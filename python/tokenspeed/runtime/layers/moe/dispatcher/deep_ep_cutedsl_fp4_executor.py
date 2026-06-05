# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Executor combining DeepEP dispatch/combine with FlashInfer CuteDSL NVFP4 MoE.

CuteDSL's grouped_gemm_nt_masked requires the [num_experts, M, hidden] layout
produced by DeepEP low-latency dispatch.  DeepEP normal dispatch returns flat
tokens that would need a separate scatter/gather path (like deep_gemm FP8),
which is incompatible with FP4 weights.
"""

import torch
from tokenspeed_kernel.ops.moe.flashinfer import (
    grouped_gemm_nt_masked,
    scaled_fp4_grouped_quantize,
    silu_and_mul_scaled_nvfp4_experts_quantize,
)

from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.moe.config import EPConfig
from tokenspeed.runtime.layers.moe.dispatcher.deep_ep import (
    DeepEPDispatcher,
)
from tokenspeed.runtime.layers.moe.dispatcher.deep_ep import (
    DeepEPMode as DispatcherDeepEPMode,
)
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.env import global_server_args_dict

logger = get_colorful_logger(__name__)


def _get_cute_dtype(t: torch.Tensor) -> str:
    if t.dtype == torch.bfloat16:
        return "bfloat16"
    if t.dtype == torch.float16:
        return "float16"
    if t.dtype == torch.float32:
        return "float32"
    raise ValueError(f"Unsupported cute dtype {t.dtype}")


def cutedsl_moe_masked(
    hidden_states: tuple[torch.Tensor, torch.Tensor | None],
    input_global_scale: torch.Tensor | None,
    w1: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alpha: torch.Tensor,
    w2: torch.Tensor,
    a2_global_scale: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alpha: torch.Tensor,
    masked_m: torch.Tensor,
):
    """Masked grouped MoE: DeepEP low-latency dispatch output -> CuteDSL NVFP4 GEMM."""
    if input_global_scale is not None and not input_global_scale.is_contiguous():
        input_global_scale = input_global_scale.contiguous()
    if not a2_global_scale.is_contiguous():
        a2_global_scale = a2_global_scale.contiguous()

    n = w2.shape[-1] * 2

    if hidden_states[1] is not None:
        a_q = hidden_states[0].view(torch.uint8)
        a_q_sf = hidden_states[1].view(torch.float8_e4m3fn)
        m, k_by_2, num_experts = a_q.shape
        k = k_by_2 * 2
    else:
        num_experts, m, k = hidden_states[0].shape
        a_q, a_q_sf = scaled_fp4_grouped_quantize(
            hidden_states[0],
            masked_m,
            input_global_scale,
        )

    sf_vec_size = 16
    ab_dtype = "float4_e2m1fn"
    sf_dtype = "float8_e4m3fn"
    c_dtype = "bfloat16"

    gateup_output = torch.empty(
        (num_experts, m, n * 2), dtype=torch.bfloat16, device=a_q.device
    )
    gateup_output = gateup_output.permute(1, 2, 0)
    grouped_gemm_nt_masked(
        (a_q, a_q_sf),
        (w1.permute(1, 2, 0), w1_blockscale),
        gateup_output,
        masked_m,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        sf_vec_size=sf_vec_size,
        alpha=w1_alpha.view(1, 1, num_experts),
        alpha_dtype=_get_cute_dtype(w1_alpha),
    )

    diq, diq_sf = silu_and_mul_scaled_nvfp4_experts_quantize(
        gateup_output.permute(2, 0, 1),
        masked_m,
        a2_global_scale,
    )

    out = torch.empty((num_experts, m, k), dtype=torch.bfloat16, device=a_q.device)
    out = out.permute(1, 2, 0)
    grouped_gemm_nt_masked(
        (diq, diq_sf),
        (w2.permute(1, 2, 0), w2_blockscale),
        out,
        masked_m,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        sf_vec_size=sf_vec_size,
        alpha=w2_alpha.view(1, 1, num_experts),
        alpha_dtype=_get_cute_dtype(w2_alpha),
    )

    return out.permute(2, 0, 1)


class DeepEPCuteDslFp4Executor:
    """DeepEP low-latency dispatch/combine + FlashInfer CuteDSL NVFP4 grouped MoE.

    All forward passes (prefill and decode) use DeepEP low-latency mode, which
    produces the ``[num_experts, M_padded, hidden]`` layout that CuteDSL's
    ``grouped_gemm_nt_masked`` requires.  This matches the low-latency FP4 path
    where FP4 cutedsl always runs with ``--deepep-mode low_latency``.
    """

    reduce_results: bool = True

    def __init__(
        self,
        top_k: int,
        num_experts: int,
        ep_rank: int,
        ep_size: int,
        hidden_size: int,
    ):
        self.top_k = top_k
        self.num_experts = num_experts
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.num_local_experts = num_experts // ep_size
        self.hidden_size = hidden_size
        self._dispatcher = None
        self._ep_group = None

    def _get_dispatcher(self) -> DeepEPDispatcher:
        if self._dispatcher is not None:
            return self._dispatcher

        mapping = global_server_args_dict["mapping"]
        config = EPConfig(
            top_k=self.top_k,
            num_experts=self.num_experts,
            low_latency_max_num_tokens_per_gpu=global_server_args_dict[
                "low_latency_max_num_tokens_per_gpu"
            ],
            max_num_tokens_per_gpu=global_server_args_dict["chunked_prefill_size"]
            // mapping.attn.tp_size,
            hidden_size=self.hidden_size,
            rank=mapping.moe.ep_rank,
            world_size=mapping.moe.ep_size,
            group=pg_manager.get_process_group("nccl", mapping.moe.tp_ep_group),
            params_dtype=torch.bfloat16,
        )
        self._dispatcher = DeepEPDispatcher(
            config,
            deepep_mode=DispatcherDeepEPMode.low_latency,
            async_finish=False,
            return_recv_hook=True,
            use_fp8=False,
        )
        self._ep_group = config.group
        return self._dispatcher

    def prewarm_low_latency(self, num_tokens: int = 16) -> None:
        dispatcher = self._get_dispatcher()
        if self._ep_group is None:
            return

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        if self.ep_rank == 0:
            hidden_states = torch.randn(
                num_tokens, self.hidden_size, dtype=torch.bfloat16, device=device
            )
            topk_ids = torch.randint(
                0,
                self.num_experts,
                (num_tokens, self.top_k),
                dtype=torch.int64,
                device=device,
            )
            topk_weights = torch.rand(
                num_tokens, self.top_k, dtype=torch.float32, device=device
            )
        else:
            hidden_states = torch.zeros(
                num_tokens, self.hidden_size, dtype=torch.bfloat16, device=device
            )
            topk_ids = torch.arange(
                self.top_k, dtype=torch.int64, device=device
            ).repeat(num_tokens, 1)
            topk_weights = torch.ones(
                num_tokens, self.top_k, dtype=torch.float32, device=device
            )

        torch.cuda.synchronize()
        torch.distributed.barrier(
            group=self._ep_group,
            device_ids=[torch.cuda.current_device()],
        )
        dispatcher.dispatch_a(
            hidden_states,
            topk_ids,
            topk_weights,
            ForwardMode.DECODE,
        )
        recv_hidden, *_ = dispatcher.dispatch_b()
        dummy_output = torch.zeros_like(recv_hidden)
        dispatcher.combine_a(
            dummy_output,
            topk_ids,
            topk_weights,
            ForwardMode.DECODE,
        )
        dispatcher.combine_b()
        torch.cuda.synchronize()
        torch.distributed.barrier(
            group=self._ep_group,
            device_ids=[torch.cuda.current_device()],
        )

    def forward(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        topk_output,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ) -> torch.Tensor:
        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights
        dispatcher = self._get_dispatcher()

        # Always use DECODE forward mode → low-latency dispatch path.
        deepep_forward_mode = ForwardMode.DECODE

        # --- Dispatch (low-latency) ---
        dispatcher.dispatch_a(
            hidden_states,
            topk_ids,
            topk_weights,
            deepep_forward_mode,
        )
        (
            recv_hidden,
            _recv_topk_ids,
            _recv_topk_weights,
            _reorder_topk_ids,
            _num_recv_tokens_per_expert_list,
            _seg_indptr,
            masked_m,
        ) = dispatcher.dispatch_b()

        # --- CuteDSL GEMM ---
        output = cutedsl_moe_masked(
            hidden_states=(recv_hidden, None),
            input_global_scale=layer.w13_input_scale_quant.expand(
                self.num_local_experts
            ),
            w1=layer.w13_weight,
            w1_blockscale=layer.w13_blockscale_swizzled,
            w1_alpha=layer.g1_alphas,
            w2=layer.w2_weight,
            a2_global_scale=layer.w2_input_scale_quant.expand(self.num_local_experts),
            w2_blockscale=layer.w2_blockscale_swizzled,
            w2_alpha=layer.g2_alphas,
            masked_m=masked_m,
        )

        # --- Combine (low-latency) ---
        dispatcher.combine_a(
            output,
            topk_ids,
            topk_weights,
            deepep_forward_mode,
        )
        return dispatcher.combine_b()
