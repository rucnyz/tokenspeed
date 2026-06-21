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

"""ctypes bindings for CUDA virtual memory management (cuMem*).

This module wraps the low-level CUDA driver virtual-memory API used by the
HiMA inter-pool capacity transfer (XPool fire).  The full lifecycle is::

    handle = cu_mem_create(size, device)        # physical allocation
    base   = cu_mem_address_reserve(size)        # virtual address window
    cu_mem_map(base, size, handle)               # bind physical -> virtual
    cu_mem_set_access(base, size, device)        # make it RW from the device
    ...                                          # use the memory
    cu_mem_unmap(base, size)                     # release the binding
    cu_mem_release(handle)                       # free physical pages
    cu_mem_address_free(base, size)              # free virtual window

All driver entry points return a ``CUresult`` (0 == success); non-zero codes
are surfaced as ``RuntimeError``.
"""

from __future__ import annotations

import ctypes
from typing import Any

import torch

_CUDA_DRIVER = None
_CHUNK_SIZE = 2 * 1024 * 1024

# CUDA driver enum values (cuda.h).
_CU_MEM_ALLOCATION_TYPE_PINNED = 1
_CU_MEM_LOCATION_TYPE_DEVICE = 1
_CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
_CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0


class _CUmemLocation(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("id", ctypes.c_int),
    ]


class _CUmemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _CUmemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class _CUmemAccessDesc(ctypes.Structure):
    _fields_ = [
        ("location", _CUmemLocation),
        ("flags", ctypes.c_int),
    ]


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


def _make_prop(device: int) -> _CUmemAllocationProp:
    prop = _CUmemAllocationProp()
    prop.type = _CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = _CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device
    return prop


def cu_mem_get_granularity(device: int) -> int:
    """Return the minimum allocation granularity (bytes) for ``device``."""
    driver = _load_driver()
    prop = _make_prop(device)
    granularity = ctypes.c_size_t(0)
    err = driver.cuMemGetAllocationGranularity(
        ctypes.byref(granularity),
        ctypes.byref(prop),
        ctypes.c_int(_CU_MEM_ALLOC_GRANULARITY_MINIMUM),
    )
    if err != 0:
        raise RuntimeError(f"cuMemGetAllocationGranularity failed: {err}")
    return int(granularity.value)


def cu_mem_create(size: int, device: int) -> int:
    """Allocate ``size`` bytes of physical device memory; return the handle."""
    driver = _load_driver()
    prop = _make_prop(device)
    handle = ctypes.c_uint64(0)
    err = driver.cuMemCreate(
        ctypes.byref(handle),
        ctypes.c_size_t(size),
        ctypes.byref(prop),
        ctypes.c_uint64(0),
    )
    if err != 0:
        raise RuntimeError(f"cuMemCreate failed: {err}")
    return int(handle.value)


def cu_mem_release(handle: int) -> None:
    """Free a physical allocation previously returned by :func:`cu_mem_create`."""
    driver = _load_driver()
    err = driver.cuMemRelease(ctypes.c_uint64(handle))
    if err != 0:
        raise RuntimeError(f"cuMemRelease failed: {err}")


def cu_mem_set_access(ptr: int, size: int, device: int) -> None:
    """Grant read/write access to ``[ptr, ptr+size)`` from ``device``."""
    driver = _load_driver()
    desc = _CUmemAccessDesc()
    desc.location.type = _CU_MEM_LOCATION_TYPE_DEVICE
    desc.location.id = device
    desc.flags = _CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    err = driver.cuMemSetAccess(
        ctypes.c_uint64(ptr),
        ctypes.c_size_t(size),
        ctypes.byref(desc),
        ctypes.c_size_t(1),
    )
    if err != 0:
        raise RuntimeError(f"cuMemSetAccess failed: {err}")


def cu_mem_address_free(ptr: int, size: int) -> None:
    """Release a virtual address window reserved by cu_mem_address_reserve."""
    driver = _load_driver()
    err = driver.cuMemAddressFree(ctypes.c_uint64(ptr), ctypes.c_size_t(size))
    if err != 0:
        raise RuntimeError(f"cuMemAddressFree failed: {err}")


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
