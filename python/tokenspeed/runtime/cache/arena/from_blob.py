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

"""Construct torch tensors from a stable VA pointer.

For CUDA device pointers we use the CUDA Array Interface (CAI) protocol so
that we never rely on private PyTorch internals.  For CPU pointers we fall
back to ``ctypes`` + ``torch.frombuffer``.
"""

from __future__ import annotations

import ctypes

import torch

# Mapping from torch dtype to the NumPy-style typestring used by CAI.
_TORCH_DTYPE_TO_TYPESTR: dict[torch.dtype, str] = {
    torch.uint8: "|u1",
    torch.int8: "|i1",
    torch.int16: "<i2",
    torch.int32: "<i4",
    torch.int64: "<i8",
    torch.float16: "<f2",
    torch.float32: "<f4",
    torch.float64: "<f8",
    # bfloat16 is not a NumPy type; represent as a 2-byte void.
    torch.bfloat16: "|V2",
}


class _CUDAArrayWrapper:
    """Minimal CUDA Array Interface v2 wrapper around a raw device pointer.

    PyTorch (and cupy / numba) recognise objects that expose
    ``__cuda_array_interface__`` and can construct zero-copy tensors from them
    via ``torch.as_tensor``.
    """

    __slots__ = ("__cuda_array_interface__",)

    def __init__(self, ptr: int, shape: tuple[int, ...], dtype: torch.dtype) -> None:
        typestr = _TORCH_DTYPE_TO_TYPESTR.get(dtype)
        if typestr is None:
            raise TypeError(
                f"from_blob: dtype {dtype} has no CUDA Array Interface typestr mapping"
            )
        self.__cuda_array_interface__ = {
            "version": 2,
            "shape": tuple(shape),
            "typestr": typestr,
            "data": (ptr, False),  # (ptr, read_only)
            "strides": None,
        }


def from_blob(
    ptr: int,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Wrap an existing memory pointer as a ``torch.Tensor`` without copying.

    For CUDA device pointers the CUDA Array Interface (CAI) protocol is used so
    PyTorch wraps the existing physical pages in-place.  For CPU pointers a
    ``ctypes`` view is used.

    The returned tensor does **not** own the underlying memory; the caller is
    responsible for keeping the VA mapping alive for as long as the tensor is in
    use.

    Args:
        ptr:    Raw integer device (or host) address.
        shape:  Desired tensor shape.
        dtype:  Desired element dtype.
        device: Target ``torch.device`` or device string.
    """
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type == "cuda":
        wrapper = _CUDAArrayWrapper(ptr, shape, dtype)
        return torch.as_tensor(wrapper, device=dev)

    # CPU path: wrap via ctypes buffer.
    numel = 1
    for d in shape:
        numel *= d
    itemsize = torch.empty((), dtype=dtype).element_size()
    buf = (ctypes.c_char * (numel * itemsize)).from_address(ptr)
    flat = torch.frombuffer(buf, dtype=dtype)
    return flat.reshape(shape)
