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

import torch
import torch.nn.functional as F
from tokenspeed_kernel._triton import tl
from tokenspeed_kernel.ops.moe.triton import (
    triton_moe_align_block_size,
    triton_moe_fused_experts,
    triton_moe_sum_reduce,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signature, format_signatures

process_weight_signature = frozenset({format_signature()})
apply_signatures = format_signatures(
    "x",
    "dense",
    {torch.float16, torch.bfloat16},
)


# ===-----------------------------------------------------------------------===#
# Un-quantized MoE
# ===-----------------------------------------------------------------------===#


@register_kernel(
    "moe_v2",
    "process_weights",
    name="triton_unquant_moe_v2_process_weights",
    solution="triton",
    signatures=process_weight_signature,
    traits={"weight_dtype": frozenset({"unquant"})},
    priority=Priority.PORTABLE,
)
def triton_unquant_moe_process_weights(plan: dict, w: torch.nn.Module):
    return None


@register_kernel(
    "moe_v2",
    "apply",
    name="triton_unquant_moe_v2_apply",
    solution="triton",
    signatures=apply_signatures,
    traits={
        "weight_dtype": frozenset({"unquant"}),
        "support_routing": frozenset({False}),
        "supports_deferred_finalize": frozenset({False}),
    },
    priority=Priority.PORTABLE,
)
def triton_unquant_moe_apply(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    router_logits: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    num_tokens_global: int | None = None,
    max_num_tokens_per_gpu: int | None = None,
):
    return triton_moe_apply_common(
        plan,
        x,
        w,
        router_logits,
        topk_weights,
        topk_ids,
        use_fp8_w8a8=False,
        block_shape=None,
    )


# ===-----------------------------------------------------------------------===#
# FP8 MoE
# ===-----------------------------------------------------------------------===#


@register_kernel(
    "moe_v2",
    "process_weights",
    name="triton_fp8_moe_v2_process_weights",
    solution="triton",
    signatures=process_weight_signature,
    traits={"weight_dtype": frozenset({"fp8"})},
    priority=Priority.PORTABLE,
)
def triton_fp8_moe_process_weights(plan: dict, w: torch.nn.Module):
    return None


@register_kernel(
    "moe_v2",
    "apply",
    name="triton_fp8_moe_v2_apply",
    solution="triton",
    signatures=apply_signatures,
    traits={
        "weight_dtype": frozenset({"fp8"}),
        "support_routing": frozenset({False}),
        "supports_deferred_finalize": frozenset({False}),
    },
    priority=Priority.PORTABLE,
)
def triton_fp8_moe_apply(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    router_logits: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    num_tokens_global: int | None = None,
    max_num_tokens_per_gpu: int | None = None,
):
    return triton_moe_apply_common(
        plan,
        x,
        w,
        router_logits,
        topk_weights,
        topk_ids,
        use_fp8_w8a8=True,
        block_shape=(128, 128),
    )


def triton_moe_apply_common(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    router_logits: torch.Tensor,
    topk_weights: torch.Tensor | None,
    topk_ids: torch.Tensor | None,
    *,
    use_fp8_w8a8: bool,
    block_shape: tuple[int, int] | None,
):
    if x.shape[0] == 0:
        return x.new_empty(0, x.shape[1])

    top_k = getattr(w, "top_k")
    if topk_weights is None or topk_ids is None:
        scores = torch.softmax(router_logits.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(scores, k=top_k, dim=-1, sorted=False)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(x.dtype)

    ep_size = getattr(w, "ep_size", 1)
    if ep_size > 1:
        num_local_experts = getattr(w, "num_local_experts", w.w13_weight.shape[0])
        local_start = getattr(w, "ep_rank", 0) * num_local_experts
        local_end = local_start + num_local_experts
        local_mask = (topk_ids >= local_start) & (topk_ids < local_end)
        topk_ids = torch.where(
            local_mask, topk_ids - local_start, torch.zeros_like(topk_ids)
        )
        topk_weights = torch.where(
            local_mask, topk_weights, torch.zeros_like(topk_weights)
        )

    num_experts, intermediate_size_x2, hidden_size = w.w13_weight.shape
    intermediate_size = intermediate_size_x2 // 2
    block_size_m = 64
    if use_fp8_w8a8:
        config = {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 32,
            "num_warps": 4,
            "num_stages": 3,
        }
    else:
        config = {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }

    sorted_token_ids, expert_ids, num_tokens_post_padded = triton_moe_align_block_size(
        topk_ids,
        block_size_m,
        num_experts,
    )
    intermediate_cache1 = torch.empty(
        (x.shape[0] * top_k, intermediate_size_x2),
        device=x.device,
        dtype=x.dtype,
    )
    intermediate_cache2 = torch.empty(
        (x.shape[0] * top_k, intermediate_size),
        device=x.device,
        dtype=x.dtype,
    )
    intermediate_cache3 = torch.empty(
        (x.shape[0], top_k, hidden_size),
        device=x.device,
        dtype=x.dtype,
    )

    triton_moe_fused_experts(
        A=x,
        B=w.w13_weight,
        bias=getattr(w, "w13_weight_bias", None),
        C=intermediate_cache1,
        A_scale=None,
        B_scale=getattr(w, "w13_weight_scale_inv", None),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=False,
        top_k=top_k,
        config=dict(config),
        compute_type=tl.bfloat16,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=block_shape,
        filter_expert=True,
    )

    gate, up = intermediate_cache1.split(intermediate_size, dim=-1)
    intermediate_cache2.copy_(F.silu(gate) * up)

    triton_moe_fused_experts(
        A=intermediate_cache2,
        B=w.w2_weight,
        bias=getattr(w, "w2_weight_bias", None),
        C=intermediate_cache3,
        A_scale=None,
        B_scale=getattr(w, "w2_weight_scale_inv", None),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=True,
        top_k=1,
        config=dict(config),
        compute_type=tl.bfloat16,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=block_shape,
        filter_expert=True,
    )

    output = torch.empty_like(x)
    triton_moe_sum_reduce(intermediate_cache3, output, 1.0)
    return output


__all__ = []
