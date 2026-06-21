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

"""Real CUDA VMM round-trip tests for the HiMA arena substrate.

Skipped automatically when no CUDA device is present.  Validates that the
cuMem* primitives (create / reserve / map / set_access / unmap / release /
address_free) actually work on the running GPU, and that a ChunkArena window
backed by real handles is readable/writable through a torch tensor.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required for VMM arena"
)


def test_vmm_primitive_roundtrip() -> None:
    from tokenspeed.runtime.cache.arena._cuda_vmm import (
        cu_mem_address_free,
        cu_mem_address_reserve,
        cu_mem_create,
        cu_mem_get_granularity,
        cu_mem_map,
        cu_mem_release,
        cu_mem_set_access,
        cu_mem_unmap,
    )

    device = torch.cuda.current_device()
    gran = cu_mem_get_granularity(device)
    assert gran > 0
    size = gran  # one granule

    handle = cu_mem_create(size, device)
    base = cu_mem_address_reserve(size, alignment=gran)
    mapped = False
    try:
        cu_mem_map(base, size, handle)
        mapped = True
        cu_mem_set_access(base, size, device)

        # Touch the memory through a torch tensor built on the VA pointer.
        from tokenspeed.runtime.cache.arena.from_blob import from_blob

        t = from_blob(base, (256,), dtype=torch.int32, device=f"cuda:{device}")
        t.fill_(7)
        torch.cuda.synchronize()
        assert int(t[0].item()) == 7
        assert int(t[-1].item()) == 7
    finally:
        # cuMemAddressFree requires the VA window to be fully unmapped first.
        if mapped:
            cu_mem_unmap(base, size)
        cu_mem_address_free(base, size)
        cu_mem_release(handle)


def test_chunk_arena_real_grow_shrink() -> None:
    from tokenspeed.runtime.cache.arena.chunk_arena import ChunkArena
    from tokenspeed.runtime.cache.arena.shared_pool import SharedHandlePool

    device = torch.cuda.current_device()
    handles = SharedHandlePool(num_chunks=4, device=device)
    arena = ChunkArena(
        shared_pool=handles, max_chunks=4, mapped_chunks=1, name="test_kv"
    )
    try:
        assert arena.mapped_chunks == 1
        grown = arena.grow(2)
        assert grown == [1, 2]
        assert arena.mapped_chunks == 3

        # Transfer round-trip: unmap then re-map a chunk.
        arena.transfer_out([2])
        assert 2 not in arena._mapped
        arena.transfer_in([2])
        assert 2 in arena._mapped

        released = arena.shrink(2)
        assert arena.mapped_chunks == 1
        assert released  # at least one chunk released
    finally:
        arena.close()
        handles.release_all()
