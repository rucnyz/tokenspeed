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
from tokenspeed_kernel.ops.moe.gluon import gluon_mxfp4_fp8_fused_moe
from tokenspeed_kernel.ops.moe.triton_kernels import (
    FP4,
    FlexCtx,
    InFlexData,
    PrecisionConfig,
    convert_layout,
    layout,
    opt_flags,
    wrap_torch_tensor,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signature, format_signatures

platform = current_platform()
process_weight_signature = frozenset({format_signature()})
apply_signatures = format_signatures(
    "x",
    "dense",
    {torch.float16, torch.bfloat16},
)


# ===-----------------------------------------------------------------------===#
# MXFP4 MoE
# ===-----------------------------------------------------------------------===#

if platform.is_amd and platform.is_cdna4:

    @register_kernel(
        "moe_v2",
        "process_weights",
        name="gluon_gfx950_mxfp4_moe_v2_process_weights",
        solution="gluon_gfx950",
        signatures=process_weight_signature,
        traits={"weight_dtype": frozenset({"mxfp4"})},
        priority=Priority.SPECIALIZED,
    )
    def gluon_gfx950_mxfp4_moe_process_weights(plan: dict, w: torch.nn.Module):
        block_size = 32
        num_warps = 8
        if layout is None:
            raise RuntimeError("triton_kernels backend unavailable")

        value_layout = layout.make_default_matmul_mxfp4_w_layout(mx_axis=-2)
        scale_layout = layout.make_default_matmul_mxfp4_w_scale_layout(
            mx_axis=-2, num_warps=num_warps
        )
        opt_flags.update_opt_flags_constraints({"block_k": 256})

        if hasattr(w, "w13_weight_bias"):
            w.w13_weight_bias = torch.nn.Parameter(
                w.w13_weight_bias.to(torch.float32), requires_grad=False
            )
        if hasattr(w, "w2_weight_bias"):
            w.w2_weight_bias = torch.nn.Parameter(
                w.w2_weight_bias.to(torch.float32), requires_grad=False
            )

        w13_quant = w.w13_weight.transpose(-2, -1)
        w13_scale_tensor = w.w13_weight_scale.transpose(-2, -1)
        w13_weight = convert_layout(
            wrap_torch_tensor(w13_quant, dtype=FP4), value_layout
        )
        w13_flex = InFlexData()
        w13_scale = convert_layout(wrap_torch_tensor(w13_scale_tensor), scale_layout)

        w2_quant = w.w2_weight.transpose(-2, -1)
        w2_scale_tensor = w.w2_weight_scale.transpose(-2, -1)
        w2_weight = convert_layout(wrap_torch_tensor(w2_quant, dtype=FP4), value_layout)
        w2_flex = InFlexData()
        w2_scale = convert_layout(wrap_torch_tensor(w2_scale_tensor), scale_layout)

        w13_in_scale = (
            w.w13_input_scale.data.to(torch.float32)
            .max()
            .reshape(1)
            .to(w.w13_input_scale.device)
            .contiguous()
        )
        w2_in_scale = (
            w.w2_input_scale.data.to(torch.float32)
            .max()
            .reshape(1)
            .to(w.w2_input_scale.device)
            .contiguous()
        )
        w.w13_act_scale = w13_in_scale
        w.w2_act_scale = w2_in_scale

        fp8_dtype = current_platform().fp8e4m3fn.dtype
        w.w13_precision_config = PrecisionConfig(
            flex_ctx=FlexCtx(
                lhs_data=InFlexData(dtype=fp8_dtype, scale=w13_in_scale),
                rhs_data=w13_flex,
            ),
            b_mx_scale=w13_scale,
            b_microblock_size=block_size,
            out_dtype=torch.bfloat16,
        )
        w.w2_precision_config = PrecisionConfig(
            flex_ctx=FlexCtx(
                lhs_data=InFlexData(dtype=fp8_dtype, scale=w2_in_scale),
                rhs_data=w2_flex,
            ),
            b_mx_scale=w2_scale,
            b_microblock_size=block_size,
            out_dtype=torch.bfloat16,
        )
        w.w13_weight_triton_tensor = w13_weight
        w.w2_weight_triton_tensor = w2_weight
        return None

    @register_kernel(
        "moe_v2",
        "apply",
        name="gluon_gfx950_mxfp4_moe_v2_apply",
        solution="gluon_gfx950",
        signatures=apply_signatures,
        traits={
            "weight_dtype": frozenset({"mxfp4"}),
            "support_routing": frozenset({True}),
            "supports_deferred_finalize": frozenset({False}),
        },
        priority=Priority.SPECIALIZED,
    )
    def gluon_gfx950_mxfp4_moe_apply(
        plan: dict,
        x: torch.Tensor,
        w: torch.nn.Module,
        router_logits: torch.Tensor,
        topk_weights: torch.Tensor | None = None,
        topk_ids: torch.Tensor | None = None,
        num_tokens_global: int | None = None,
        max_num_tokens_per_gpu: int | None = None,
    ):
        swiglu_arg = getattr(w, "swiglu_arg", None)
        swiglu_alpha = swiglu_arg.alpha if swiglu_arg else 1.702
        swiglu_limit = swiglu_arg.limit if swiglu_arg else 7.0
        return gluon_mxfp4_fp8_fused_moe(
            x,
            router_logits,
            w.w13_weight_triton_tensor,
            w.w2_weight_triton_tensor,
            w13_bias=getattr(w, "w13_weight_bias", None),
            w2_bias=getattr(w, "w2_weight_bias", None),
            w13_precision_config=w.w13_precision_config,
            w2_precision_config=w.w2_precision_config,
            w13_act_scale=w.w13_act_scale,
            w2_act_scale=w.w2_act_scale,
            top_k=getattr(w, "top_k"),
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
        )


__all__ = []
