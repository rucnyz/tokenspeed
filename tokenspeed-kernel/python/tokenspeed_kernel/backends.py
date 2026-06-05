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

"""Lazy loading for built-in kernel registration modules."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterable
from threading import RLock

__all__ = ["load_builtin_kernels", "reset_builtin_kernel_load_state"]


_BUILTIN_MODULES_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "attention": (
        "tokenspeed_kernel.ops.attention.cuda",
        "tokenspeed_kernel.ops.attention.flash_attn",
        "tokenspeed_kernel.ops.attention.flashinfer",
        "tokenspeed_kernel.ops.attention.gluon",
        "tokenspeed_kernel.ops.attention.triton",
    ),
    "embedding": (
        "tokenspeed_kernel.ops.embedding.cuda",
        "tokenspeed_kernel.ops.embedding.triton",
    ),
    "gemm": (
        "tokenspeed_kernel.numerics.reference.gemm",
        "tokenspeed_kernel.ops.gemm.deep_gemm",
        "tokenspeed_kernel.ops.gemm.flashinfer",
        "tokenspeed_kernel.ops.gemm.triton",
        "tokenspeed_kernel.ops.gemm.trtllm",
    ),
    "moe": (
        "tokenspeed_kernel.numerics.reference.moe",
        "tokenspeed_kernel.ops.moe.cuda",
        "tokenspeed_kernel.ops.moe.deepep",
        "tokenspeed_kernel.ops.moe.flashinfer",
        "tokenspeed_kernel.ops.moe.gluon",
        "tokenspeed_kernel.ops.moe.triton",
        "tokenspeed_kernel.ops.moe.triton_kernels",
        "tokenspeed_kernel.ops.moe.trtllm",
    ),
    "quantization": (
        "tokenspeed_kernel.ops.quantization.flashinfer",
        "tokenspeed_kernel.ops.quantization.triton",
        "tokenspeed_kernel.ops.quantization.trtllm",
    ),
}

_LOCK = RLock()
_LOADED_FAMILIES: set[str] = set()


def _all_builtin_modules() -> tuple[str, ...]:
    modules: list[str] = []
    for family_modules in _BUILTIN_MODULES_BY_FAMILY.values():
        modules.extend(family_modules)
    return tuple(modules)


def reset_builtin_kernel_load_state() -> None:
    """Forget loaded-family state and cached registration modules."""

    with _LOCK:
        _LOADED_FAMILIES.clear()
        builtin_modules = _all_builtin_modules()
        for name in list(sys.modules):
            if any(
                name == module_name or name.startswith(module_name + ".")
                for module_name in builtin_modules
            ):
                del sys.modules[name]


def _normalize_families(families: str | Iterable[str] | None) -> tuple[str, ...]:
    if families is None:
        return tuple(_BUILTIN_MODULES_BY_FAMILY)
    if isinstance(families, str):
        return (families,)
    return tuple(families)


def load_builtin_kernels(families: str | Iterable[str] | None = None) -> None:
    """Import built-in registration modules for one or more op families.

    This is intentionally separate from importing public API packages. Built-in
    modules perform platform detection and optional vendor-library imports, so
    they are loaded only when an API call needs them or tests request them.
    """

    family_names = _normalize_families(families)

    with _LOCK:
        from tokenspeed_kernel.registry import KernelRegistry

        if not KernelRegistry.get().list_kernels():
            reset_builtin_kernel_load_state()

        for family in family_names:
            try:
                module_names = _BUILTIN_MODULES_BY_FAMILY[family]
            except KeyError as exc:
                valid = ", ".join(sorted(_BUILTIN_MODULES_BY_FAMILY))
                raise ValueError(
                    f"unknown built-in kernel family {family!r}: {valid}"
                ) from exc

            if family in _LOADED_FAMILIES:
                continue

            for module_name in module_names:
                importlib.import_module(module_name)
            _LOADED_FAMILIES.add(family)
