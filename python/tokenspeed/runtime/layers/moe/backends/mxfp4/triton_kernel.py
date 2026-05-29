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

from __future__ import annotations

import tokenspeed_kernel
import torch
from tokenspeed_kernel.ops.moe.triton_kernels import (
    FP4,
    FlexCtx,
    FnSpecs,
    FusedActivation,
    InFlexData,
    PrecisionConfig,
    convert_layout,
    layout,
    opt_flags,
    swiglu_fn,
    wrap_torch_tensor,
)
from tokenspeed_kernel.platform import current_platform
from torch import nn
from torch.nn.parameter import Parameter

from tokenspeed.runtime.layers.moe.backends.base import MoEBackend
from tokenspeed.runtime.layers.moe.backends.mxfp4.weights import (
    MXFP4_BLOCK,
    create_mxfp4_fp8_input_scales,
    create_mxfp4_weights,
)
from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec
from tokenspeed.runtime.layers.moe.topk import TopKOutputFormat
from tokenspeed.runtime.layers.quantization import Mxfp4Config
from tokenspeed.runtime.layers.quantization.utils import should_ignore_quant_layer
from tokenspeed.runtime.utils import round_up

_is_blackwell = current_platform().is_blackwell
_is_hopper = current_platform().is_hopper
_is_amd = current_platform().is_amd


def swizzle_mxfp4(quant_tensor, scale, num_warps):
    """Weight swizzle for mxfp4 MoE, used for OAI mxfp4 kernel."""

    if layout is None:
        raise RuntimeError("triton_kernels backend unavailable")

    value_layout = layout.make_default_matmul_mxfp4_w_layout(mx_axis=-2)
    scale_layout = layout.make_default_matmul_mxfp4_w_scale_layout(
        mx_axis=-2, num_warps=num_warps
    )
    if _is_blackwell:
        constraints = {
            "is_persistent": True,
            "epilogue_subtile": 1,
        }
        opt_flags.update_opt_flags_constraints(constraints)
    elif _is_hopper:
        constraints = {
            "split_k": 1,
        }
        opt_flags.update_opt_flags_constraints(constraints)
    elif _is_amd:
        # Fix block_k=256 to support scale swizzling.
        constraints = {
            "block_k": 256,
        }
        opt_flags.update_opt_flags_constraints(constraints)
    # transpose the tensor so that the quantization axis is on dim1
    quant_tensor = quant_tensor.transpose(-2, -1)
    scale = scale.transpose(-2, -1)
    quant_tensor = convert_layout(
        wrap_torch_tensor(quant_tensor, dtype=FP4), value_layout
    )
    scale = convert_layout(wrap_torch_tensor(scale), scale_layout)
    return quant_tensor, InFlexData(), scale


def _pad_w2_to_block_n(layer: nn.Module, block_n: int) -> None:
    original_n = int(layer.w2_weight.shape[-2])
    layer._w2_logical_n = original_n

    if original_n % block_n == 0:
        return

    n_padded = (original_n + block_n - 1) // block_n * block_n
    extra_n = n_padded - original_n

    w2_w = layer.w2_weight.data
    w2_s = layer.w2_weight_scale.data
    w2_w_padded = torch.cat(
        [
            w2_w,
            torch.zeros(
                *w2_w.shape[:-2],
                extra_n,
                w2_w.shape[-1],
                dtype=w2_w.dtype,
                device=w2_w.device,
            ),
        ],
        dim=-2,
    )
    w2_s_padded = torch.cat(
        [
            w2_s,
            torch.full(
                (*w2_s.shape[:-2], extra_n, w2_s.shape[-1]),
                127,
                dtype=w2_s.dtype,
                device=w2_s.device,
            ),
        ],
        dim=-2,
    )
    layer.w2_weight = Parameter(w2_w_padded, requires_grad=False)
    layer.w2_weight_scale = Parameter(w2_s_padded, requires_grad=False)


def _attach_gluon_bpreshuffle(layer: nn.Module) -> None:
    if not _is_amd:
        return

    try:
        from tokenspeed_kernel.ops.moe.gluon import (
            _extract_gluon_raw_w,
            shuffle_weight_for_gluon_dot_layout,
        )
    except ImportError:
        return

    targets = (
        ("w13_weight_triton_tensor", None),
        ("w2_weight_triton_tensor", getattr(layer, "_w2_logical_n", None)),
    )
    for attr, logical_n in targets:
        wrapped = getattr(layer, attr, None)
        if wrapped is None:
            continue
        w_raw = _extract_gluon_raw_w(wrapped)
        try:
            w_shuffled = shuffle_weight_for_gluon_dot_layout(w_raw)
        except (ValueError, AssertionError):
            continue
        if logical_n is not None and logical_n != w_shuffled.shape[-1]:
            # combine GEMM: backend pre-padded N to a multiple of BLOCK_N.
            # Stamp the logical N on BOTH the shuffled tensor (consumed by
            # ``gluon_mxfp_combine`` on the W_VIA_VGPR path) and the raw
            # tensor (consumed by the LDS fallback when BLOCK_M<=32).
            w_shuffled.original_n = int(logical_n)
            w_raw.original_n = int(logical_n)
        w_raw._gluon_shuffled = w_shuffled


class Mxfp4TritonKernelBackend(MoEBackend):
    supported_arches = frozenset({"any"})

    def __init__(
        self,
        key,
        spec: MoELayerSpec,
        quant_config: object,
        routing_config: dict | None = None,
    ):
        del routing_config
        self.key = key
        self.spec = spec
        self.quant_config = quant_config
        self._activation: str | None = None
        self._swiglu_arg = None
        self._is_w4a8_fp8 = (
            isinstance(quant_config, Mxfp4Config)
            and quant_config.is_w4a8_fp8
            and current_platform().is_amd
        )

    @classmethod
    def supports(cls, spec: MoELayerSpec, quant_config: object) -> bool:
        if not isinstance(quant_config, Mxfp4Config):
            return False
        if should_ignore_quant_layer(
            prefix=spec.prefix,
            ignored_layers=getattr(quant_config, "ignored_layers", []) or [],
        ):
            return False
        if quant_config.is_w4a8_fp8:
            if not current_platform().is_amd:
                # Quark quantization has only been tested on AMD platform
                return False
        return spec.ep_size <= 1 and spec.activation in {
            "silu",
            "swiglu",
        }

    @property
    def topk_output_format(self) -> TopKOutputFormat:
        return TopKOutputFormat.BYPASSED

    def create_layer_weights(
        self, layer: nn.Module, *, with_bias: bool = False
    ) -> None:

        hidden = self.spec.hidden_size
        ispp = self.spec.intermediate_size // self.spec.tp_size

        if current_platform().is_blackwell:
            ispp_padded = round_up(ispp, 64)
        else:
            ispp_padded = round_up(ispp, MXFP4_BLOCK)
        hidden_padded = hidden

        create_mxfp4_weights(
            self,
            layer,
            self.spec.num_local_experts,
            hidden_padded,
            ispp_padded,
            with_bias=with_bias,
        )
        if self._is_w4a8_fp8:
            create_mxfp4_fp8_input_scales(layer, self.spec.num_local_experts)

        self._activation = layer.activation
        self._swiglu_arg = getattr(layer, "swiglu_arg", None)

    def process_weights_after_loading(self, layer: nn.Module) -> None:

        MXFP_BLOCK_SIZE = 32

        w13_weight_bias = layer.w13_weight_bias.to(torch.float32)
        w2_weight_bias = layer.w2_weight_bias.to(torch.float32)
        layer.w13_weight_bias = Parameter(w13_weight_bias, requires_grad=False)
        layer.w2_weight_bias = Parameter(w2_weight_bias, requires_grad=False)

        # Pre-pad w2 along N before swizzle so the gluon W_VIA_VGPR path can
        # drop its n-mask. Only relevant on AMD where bpreshuffle is wired up.
        if self._is_w4a8_fp8 and _is_amd:
            _GLUON_COMBINE_BLOCK_N = 128
            _pad_w2_to_block_n(layer, _GLUON_COMBINE_BLOCK_N)

        num_warps = 8
        w13_weight, w13_flex, w13_scale = swizzle_mxfp4(
            layer.w13_weight, layer.w13_weight_scale, num_warps
        )
        w2_weight, w2_flex, w2_scale = swizzle_mxfp4(
            layer.w2_weight, layer.w2_weight_scale, num_warps
        )

        if self._is_w4a8_fp8:
            # Collapse per-expert input scales to a single per-tensor scale
            # per GEMM. Quark exports a constant value across experts for
            # static ``per_tensor`` quantisation; ``max`` is a safe reduction
            # in case individual experts reach slightly different values.
            w13_in_scale = (
                layer.w13_input_scale.data.to(torch.float32)
                .max()
                .reshape(1)
                .to(layer.w13_input_scale.device)
                .contiguous()
            )
            w2_in_scale = (
                layer.w2_input_scale.data.to(torch.float32)
                .max()
                .reshape(1)
                .to(layer.w2_input_scale.device)
                .contiguous()
            )
            layer.w13_act_scale = w13_in_scale
            layer.w2_act_scale = w2_in_scale

            fp8_dtype = current_platform().fp8e4m3fn.dtype
            w13_lhs = InFlexData(dtype=fp8_dtype, scale=w13_in_scale)
            w2_lhs = InFlexData(dtype=fp8_dtype, scale=w2_in_scale)
            # Force bf16 output so the swiglu / down-proj results stay in a
            # standard floating dtype; without this, ``triton_kernels.matmul``
            # defaults ``out_dtype`` to the input dtype (fp8) which would
            # make the subsequent reductions / re-quantisation blow up.
            out_dtype = torch.bfloat16
        else:
            w13_lhs = InFlexData()
            w2_lhs = InFlexData()
            out_dtype = None

        layer.w13_precision_config = PrecisionConfig(
            flex_ctx=FlexCtx(lhs_data=w13_lhs, rhs_data=w13_flex),
            b_mx_scale=w13_scale,
            b_microblock_size=MXFP_BLOCK_SIZE,
            out_dtype=out_dtype,
        )
        layer.w2_precision_config = PrecisionConfig(
            flex_ctx=FlexCtx(lhs_data=w2_lhs, rhs_data=w2_flex),
            b_mx_scale=w2_scale,
            b_microblock_size=MXFP_BLOCK_SIZE,
            out_dtype=out_dtype,
        )

        layer.w13_weight_triton_tensor = w13_weight
        layer.w2_weight_triton_tensor = w2_weight
        # Free original weights (replaced by shuffled versions)
        del layer.w13_weight
        del layer.w2_weight

        # AMD bpreshuffle: stamp gluon dot-layout W on the wrapped tensors so
        # the dispatch / combine GEMMs can take the W_VIA_VGPR fast path.
        if self._is_w4a8_fp8:
            _attach_gluon_bpreshuffle(layer)

        torch.cuda.empty_cache()

    def forward(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        topk_output: object,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ) -> torch.Tensor:
        del num_global_tokens, max_num_tokens_per_gpu
        router_logits = topk_output.router_logits
        top_k = topk_output.topk_config.top_k
        n_tokens = router_logits.shape[0]

        ragged_metadata, gather_indx, scatter_indx, gate_scal = (
            tokenspeed_kernel.moe_route(
                router_logits,
                top_k,
                sm_first=False,
                dtype=router_logits.dtype,
                traits={"output_type": "ragged_metadata"},
                expected_kernel_name="triton_kernels_routing",
            )
        )

        w13_weight = layer.w13_weight_triton_tensor
        w2_weight = layer.w2_weight_triton_tensor
        w13_bias = getattr(layer, "w13_weight_bias", None)
        w2_bias = getattr(layer, "w2_weight_bias", None)
        w13_pc = getattr(layer, "w13_precision_config", None)
        w2_pc = getattr(layer, "w2_precision_config", None)

        gemm1_alpha = self._swiglu_arg.alpha if self._swiglu_arg else 1.702
        gemm1_clamp = self._swiglu_arg.limit if self._swiglu_arg else 7.0

        act = FusedActivation(
            FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2),
            (gemm1_alpha, gemm1_clamp),
        )

        if self._is_w4a8_fp8:
            gemm1_input = tokenspeed_kernel.quantize_fp8(
                hidden_states,
                scale=layer.w13_act_scale,
                solution="triton",
            )
            # Fused output quant on the SwiGLU epilogue: writes FP8 directly so
            # the second GEMM can skip a separate quantise pass on the gluon
            # path. ``traits={"weight_dtype": "mxfp4"}`` selects the gluon
            # dispatch kernel.
            gemm1_out_quant_scale = layer.w2_act_scale
            gemm1_traits = {"weight_dtype": "mxfp4"}
            gemm2_traits = {"weight_dtype": "mxfp4"}
        else:
            gemm1_input = hidden_states
            gemm1_out_quant_scale = None
            gemm1_traits = None
            gemm2_traits = None

        # First GEMM: gate_up projection with fused activation
        intermediate_cache = tokenspeed_kernel.moe_experts(
            gemm1_input,
            w13_weight,
            w13_bias,
            a_ragged_metadata=ragged_metadata,
            gather_indx=gather_indx,
            precision_config=w13_pc,
            fused_activation=act,
            dtype=hidden_states.dtype,
            features={"ragged_metadata", "dispatch_gemm"},
            traits=gemm1_traits,
            expected_kernel_name="triton_kernels_dispatch_gemm",
            out_quant_scale=gemm1_out_quant_scale,
        )

        if self._is_w4a8_fp8:
            # Skip a redundant quantise when the gluon dispatch GEMM already
            # produced FP8 via the fused ``out_quant_scale`` epilogue.
            if intermediate_cache.dtype in (
                torch.float8_e4m3fn,
                torch.float8_e4m3fnuz,
            ):
                gemm2_input = intermediate_cache
            else:
                gemm2_input = tokenspeed_kernel.quantize_fp8(
                    intermediate_cache,
                    scale=layer.w2_act_scale,
                    solution="triton",
                )
        else:
            gemm2_input = intermediate_cache

        # Second GEMM: down projection with scatter (combine)
        # gammas applies the routing weights (expert contribution weights)
        return tokenspeed_kernel.moe_experts(
            gemm2_input,
            w2_weight,
            w2_bias,
            a_ragged_metadata=ragged_metadata,
            scatter_indx=scatter_indx,
            precision_config=w2_pc,
            gammas=gate_scal,
            n_tokens=n_tokens,
            n_expts_act=top_k,
            dtype=hidden_states.dtype,
            features={"ragged_metadata", "gemm_combine"},
            traits=gemm2_traits,
            expected_kernel_name="triton_kernels_gemm_combine",
        )


__all__ = ["Mxfp4TritonKernelBackend"]
