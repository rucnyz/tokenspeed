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
# Backend registration (side-effect imports)
import tokenspeed_kernel.ops.moe_v2.flashinfer  # noqa: F401
import tokenspeed_kernel.ops.moe_v2.gluon  # noqa: F401
import tokenspeed_kernel.ops.moe_v2.triton  # noqa: F401
import torch
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = [
    "moe_apply",
    "moe_plan",
    "moe_process_weights",
]


def moe_plan(
    weight_dtype: str,
    solution: str | None = None,
    *,
    a2a_backend: str | None = None,
    deepep_group: object | None = None,
) -> dict:
    """Create a MoE v2 execution plan.

    Args:
        weight_dtype: Logical MoE weight dtype. fp16, bf16, float16,
            bfloat16, and unquantized aliases map to unquant.
        solution: Optional kernel solution to force through normal selection.
            None leaves the concrete kernel choice to the registry.
        a2a_backend: Optional all-to-all backend. deepep selects the DeepEP
            solution when solution is not set.
        deepep_group: Runtime-created process group used by DeepEP plans.

    The selected apply kernel owns plan metadata. A plan with support_routing
    false requires precomputed top-k ids and weights when calling moe_apply.
    Process-weights follows the selected apply solution when weights are loaded.
    """
    if weight_dtype in {"bf16", "fp16", "float16", "bfloat16", "unquantized"}:
        weight_dtype = "unquant"
    if solution is None and a2a_backend == "deepep":
        solution = "flashinfer_cutedsl_deepep"

    kernel = select_kernel(
        "moe_v2",
        "apply",
        format_signature(x=dense_tensor_format(torch.bfloat16)),
        traits={"weight_dtype": weight_dtype},
        solution=solution,
    )
    spec = KernelRegistry.get().get_by_name(kernel.name)
    if spec is None:
        raise RuntimeError(f"Kernel spec not found for selected kernel {kernel.name}")
    support_routing = True in spec.traits.get("support_routing", frozenset({False}))
    supports_deferred_finalize = True in spec.traits.get(
        "supports_deferred_finalize", frozenset({False})
    )
    return {
        "weight_dtype": weight_dtype,
        "solution": spec.solution,
        "a2a_backend": a2a_backend,
        "deepep_group": deepep_group,
        "support_routing": support_routing,
        "supports_deferred_finalize": supports_deferred_finalize,
    }


def moe_process_weights(plan: dict, w: torch.nn.Module):
    """Process loaded MoE weights according to a plan.

    Args:
        plan: Execution plan returned by moe_plan.
        w: Module containing loaded MoE weights. This module is mutated in
            place to prepare solution-specific layouts and scales.
    """
    kernel = select_kernel(
        "moe_v2",
        "process_weights",
        format_signature(),
        traits={"weight_dtype": plan["weight_dtype"]},
        solution=plan["solution"],
    )
    return kernel(plan=plan, w=w)


def moe_apply(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    # top-k routing inputs
    router_logits: torch.Tensor,
    # top-k routing results
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    # token length
    num_tokens_global: int | None = None,
    max_num_tokens_per_gpu: int | None = None,
):
    """Apply a planned MoE kernel.

    Args:
        plan: Execution plan returned by moe_plan.
        x: Hidden states with shape [tokens, hidden_size].
        w: Module containing processed MoE weights.
        router_logits: Router logits with shape [tokens, num_experts].
        topk_weights: Optional precomputed expert weights with shape
            [tokens, top_k]. Required when plan support_routing is false.
        topk_ids: Optional precomputed expert ids with shape [tokens, top_k].
            Required when plan support_routing is false.
        num_tokens_global: Optional global token count for distributed MoE.
        max_num_tokens_per_gpu: Optional per-GPU token capacity hint.

    Solutions may use precomputed top-k tensors or route from logits directly.
    """
    kernel = select_kernel(
        "moe_v2",
        "apply",
        format_signature(x=dense_tensor_format(x.dtype)),
        traits={"weight_dtype": plan["weight_dtype"]},
        solution=plan["solution"],
    )
    return kernel(
        plan=plan,
        x=x,
        w=w,
        router_logits=router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        num_tokens_global=num_tokens_global,
        max_num_tokens_per_gpu=max_num_tokens_per_gpu,
    )
