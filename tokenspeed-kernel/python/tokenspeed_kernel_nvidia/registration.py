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

"""Plugin entry point for NVIDIA TokenSpeed kernels."""

from __future__ import annotations

import importlib
import sys

_REGISTRATION_MODULES = (
    "tokenspeed_kernel.ops.activation.cuda",
    "tokenspeed_kernel.ops.activation.flashinfer",
    "tokenspeed_kernel.ops.activation.triton",
    "tokenspeed_kernel.ops.attention.cuda",
    "tokenspeed_kernel.ops.attention.flash_attn",
    "tokenspeed_kernel.ops.attention.flash_mla",
    "tokenspeed_kernel.ops.attention.flashinfer",
    "tokenspeed_kernel.ops.attention.tokenspeed_mla",
    "tokenspeed_kernel.ops.communication.flashinfer",
    "tokenspeed_kernel.ops.communication.nccl",
    "tokenspeed_kernel.ops.communication.trtllm",
    "tokenspeed_kernel.ops.embedding.cuda",
    "tokenspeed_kernel.ops.embedding.flashinfer",
    "tokenspeed_kernel.ops.gemm.cute_dsl",
    "tokenspeed_kernel.ops.gemm.deep_gemm",
    "tokenspeed_kernel.ops.gemm.flashinfer",
    "tokenspeed_kernel.ops.gemm.triton",
    "tokenspeed_kernel.ops.gemm.trtllm",
    "tokenspeed_kernel.ops.kvcache.cuda",
    "tokenspeed_kernel.ops.layernorm.cuda",
    "tokenspeed_kernel.ops.layernorm.flashinfer",
    "tokenspeed_kernel.ops.moe.cuda",
    "tokenspeed_kernel.ops.moe.deepep",
    "tokenspeed_kernel.ops.moe.flashinfer",
    "tokenspeed_kernel.ops.moe.triton",
    "tokenspeed_kernel.ops.moe.triton_kernels",
    "tokenspeed_kernel.ops.moe.trtllm",
    "tokenspeed_kernel.ops.quantization.cuda",
    "tokenspeed_kernel.ops.quantization.flashinfer",
    "tokenspeed_kernel.ops.quantization.trtllm",
    "tokenspeed_kernel.ops.routing.cuda",
    "tokenspeed_kernel.ops.sampling.cuda",
    "tokenspeed_kernel.ops.sampling.cute_dsl",
    "tokenspeed_kernel.ops.sampling.flashinfer",
)


def register() -> None:
    """Register NVIDIA kernels by importing their registration modules."""

    for module_name in _REGISTRATION_MODULES:
        module = sys.modules.get(module_name)
        if module is None:
            importlib.import_module(module_name)
        else:
            importlib.reload(module)
