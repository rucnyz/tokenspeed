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

"""CUDA-based MoE helper imports."""

from __future__ import annotations

from tokenspeed_kernel.ops.routing.cuda import routing_flash as cuda_routing_flash
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import error_fn

platform = current_platform()

silu_and_mul_fuse_block_quant = error_fn
moe_finalize_fuse_shared = error_fn

if platform.is_nvidia:
    from tokenspeed_kernel.thirdparty.cuda.activation import (
        silu_and_mul_fuse_block_quant,
    )
    from tokenspeed_kernel.thirdparty.cuda.moe import moe_finalize_fuse_shared

__all__ = [
    "cuda_routing_flash",
    "moe_finalize_fuse_shared",
    "silu_and_mul_fuse_block_quant",
]
