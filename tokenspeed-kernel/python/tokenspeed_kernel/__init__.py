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

import importlib
import os
from collections.abc import Callable
from typing import Any

# Keep these names in sync with profiling.py without importing that module here;
# importing profiling would pull in _triton/proton during plain package import.
_PROFILE_BOOTSTRAP_ENVS = frozenset(
    {
        "TOKENSPEED_KERNEL_PROFILE",
        "TOKENSPEED_KERNEL_CAPTURE_SHAPES",
    }
)
_PUBLIC_APIS: dict[str, tuple[str, str]] = {
    # gemm
    "mm": ("tokenspeed_kernel.ops.gemm", "mm"),
    # moe
    "moe_route": ("tokenspeed_kernel.ops.moe", "moe_route"),
    "moe_dispatch": ("tokenspeed_kernel.ops.moe", "moe_dispatch"),
    "moe_experts": ("tokenspeed_kernel.ops.moe", "moe_experts"),
    "moe_combine": ("tokenspeed_kernel.ops.moe", "moe_combine"),
    "moe_fused": ("tokenspeed_kernel.ops.moe", "moe_fused"),
    # attention
    "mha_prefill": ("tokenspeed_kernel.ops.attention", "mha_prefill"),
    "mha_extend_with_kvcache": (
        "tokenspeed_kernel.ops.attention",
        "mha_extend_with_kvcache",
    ),
    "mha_decode_with_kvcache": (
        "tokenspeed_kernel.ops.attention",
        "mha_decode_with_kvcache",
    ),
    "mha_merge_state": ("tokenspeed_kernel.ops.attention", "mha_merge_state"),
    "mha_decode_scheduler_metadata": (
        "tokenspeed_kernel.ops.attention",
        "mha_decode_scheduler_metadata",
    ),
    # quantization
    "quantize_fp8": ("tokenspeed_kernel.ops.quantization", "quantize_fp8"),
    "quantize_fp8_with_scale": (
        "tokenspeed_kernel.ops.quantization",
        "quantize_fp8_with_scale",
    ),
    "quantize_mxfp8": ("tokenspeed_kernel.ops.quantization", "quantize_mxfp8"),
    "quantize_nvfp4": ("tokenspeed_kernel.ops.quantization", "quantize_nvfp4"),
    "quantize_mxfp4": ("tokenspeed_kernel.ops.quantization", "quantize_mxfp4"),
}

__all__ = list(_PUBLIC_APIS)


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _profiling_env_enabled() -> bool:
    return any(_is_truthy(os.environ.get(name)) for name in _PROFILE_BOOTSTRAP_ENVS)


def _bootstrap_profiling_if_requested() -> None:
    if not _profiling_env_enabled():
        return
    from tokenspeed_kernel.profiling import bootstrap_profiling_from_env

    bootstrap_profiling_from_env()


def _resolve_public_api(name: str) -> Callable[..., Any]:
    """Import and return the concrete implementation for a public API name."""
    module_name, attr_name = _PUBLIC_APIS[name]
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _make_public_api(name: str) -> Callable[..., Any]:
    """Create a thin lazy wrapper that resolves the implementation on call."""

    def public_api(*args: Any, **kwargs: Any) -> Any:
        return _resolve_public_api(name)(*args, **kwargs)

    public_api.__name__ = name
    public_api.__qualname__ = name
    public_api.__module__ = __name__
    return public_api


_bootstrap_profiling_if_requested()

for _name in __all__:
    # Publish callables without importing ops modules until the API is used.
    globals()[_name] = _make_public_api(_name)

del _name
