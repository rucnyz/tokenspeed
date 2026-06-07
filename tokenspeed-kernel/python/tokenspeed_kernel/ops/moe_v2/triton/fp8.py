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
from tokenspeed_kernel.ops.moe_v2.triton._common import triton_moe_apply_common
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signature, format_signatures

process_weight_signature = frozenset({format_signature()})
apply_signatures = format_signatures(
    "x",
    "dense",
    {torch.float16, torch.bfloat16},
)


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
