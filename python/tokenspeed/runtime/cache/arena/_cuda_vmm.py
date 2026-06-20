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

"""ctypes bindings for CUDA virtual memory management (cuMem*)."""

from __future__ import annotations

import ctypes
from typing import Any

import torch

_CUDA_DRIVER = None
_CHUNK_SIZE = 2 * 1024 * 1024


def _load_driver() -> Any:
    global _CUDA_DRIVER
    if _CUDA_DRIVER is not None:
        return _CUDA_DRIVER
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for VMM arena")
    from torch.utils.cpp_extension import CUDA_HOME  # type: ignore[attr-defined]

    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is not set; cannot load cuda driver")
    _CUDA_DRIVER = ctypes.CDLL("libcuda.so.1")
    return _CUDA_DRIVER


def cu_mem_address_reserve(size: int, alignment: int = _CHUNK_SIZE) -> int:
    driver = _load_driver()
    ptr = ctypes.c_uint64(0)
    err = driver.cuMemAddressReserve(
        ctypes.byref(ptr),
        ctypes.c_size_t(size),
        ctypes.c_size_t(alignment),
        ctypes.c_uint64(0),
        ctypes.c_uint64(0),
    )
    if err != 0:
        raise RuntimeError(f"cuMemAddressReserve failed: {err}")
    return int(ptr.value)


def cu_mem_unmap(ptr: int, size: int) -> None:
    driver = _load_driver()
    err = driver.cuMemUnmap(ctypes.c_uint64(ptr), ctypes.c_size_t(size))
    if err != 0:
        raise RuntimeError(f"cuMemUnmap failed: {err}")


def cu_mem_map(ptr: int, size: int, handle: int, offset: int = 0) -> None:
    driver = _load_driver()
    err = driver.cuMemMap(
        ctypes.c_uint64(ptr),
        ctypes.c_size_t(size),
        ctypes.c_size_t(offset),
        ctypes.c_uint64(handle),
        ctypes.c_uint64(0),
    )
    if err != 0:
        raise RuntimeError(f"cuMemMap failed: {err}")


CHUNK_SIZE_BYTES = _CHUNK_SIZE
