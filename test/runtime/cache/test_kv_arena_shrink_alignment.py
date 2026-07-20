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

"""Regression tests for :meth:`KvLayerArenaGroup.grow` / ``shrink`` alignment.

Guards the ``sys``/``sys_lru`` KV-store ``CUDA error: an illegal memory
access`` crash root-caused on 2026-07-16.  The KV arena maps/unmaps physical
VMM memory in whole ``CHUNK_SIZE_BYTES`` chunks, while the C++ page allocator
caps/frees at single-page granularity.  The old code unmapped
``ceil(pages→chunks)`` on every ``shrink`` call, so a sub-chunk ``kv_to_mamba``
fire (e.g. 8 pages with 16 pages/chunk) unmapped a *full* chunk while the C++
side only capped 8 pages.  Repeated sub-chunk fires eroded the initial slack and
eventually left C++ pages allocatable with no physical backing -- a phantom
page that faults in ``_store_kv_cache_kernel`` once the pool fills to it at true
exhaustion.  The invariant that must hold after every op is:

    arena physically-mapped pages >= C++ allocatable pages
"""

from __future__ import annotations

import math

from tokenspeed.runtime.cache.arena._cuda_vmm import CHUNK_SIZE_BYTES
from tokenspeed.runtime.cache.arena.kv_arena import KvLayerArenaGroup


class _FakeChunkArena:
    """Minimal ChunkArena stub tracking only the mapped/max chunk counts."""

    def __init__(self, name: str, mapped_chunks: int, max_chunks: int) -> None:
        self.name = name
        self.mapped_chunks = mapped_chunks
        self.max_chunks = max_chunks

    def grow(self, n: int) -> None:
        assert self.mapped_chunks + n <= self.max_chunks, "grew past max_chunks"
        self.mapped_chunks += n

    def shrink(self, n: int) -> None:
        assert self.mapped_chunks - n >= 0, "shrank below zero"
        self.mapped_chunks -= n


def _make_group(*, initial_mapped_chunks: int, max_chunks: int) -> KvLayerArenaGroup:
    # page_size * head_num * head_dim * itemsize = CHUNK_SIZE_BYTES / 16
    # => 16 pages per chunk (the production Qwen3.5-9B KV geometry).
    page_size = 64
    head_num = 8
    head_dim = 128
    dtype_itemsize = 2
    bytes_per_page = page_size * head_num * head_dim * dtype_itemsize
    assert CHUNK_SIZE_BYTES % bytes_per_page == 0
    assert CHUNK_SIZE_BYTES // bytes_per_page == 16
    return KvLayerArenaGroup(
        k_arenas=[_FakeChunkArena("k0", initial_mapped_chunks, max_chunks)],
        v_arenas=[_FakeChunkArena("v0", initial_mapped_chunks, max_chunks)],
        page_size=page_size,
        head_num=head_num,
        head_dim=head_dim,
        dtype_itemsize=dtype_itemsize,
    )


_PAGES_PER_CHUNK = 16


def _physical_pages(group: KvLayerArenaGroup) -> int:
    return group.mapped_chunks * _PAGES_PER_CHUNK


def test_subchunk_shrinks_never_unmap_below_cpp_boundary() -> None:
    """The exact crash sequence: 16×64-page fires then 3×8-page fires.

    C++ pages start at 30312 (production: ``max_total_tokens=1940000`` /
    ``block_size=64``), which is ``1894*16 + 8`` -- exactly 8 pages of slack in
    the tail chunk.  After every fire the physically-mapped page count must stay
    >= the C++ allocatable page count.
    """
    # 30312 C++ pages -> ceil(30312/16) = 1895 mapped chunks (30320, slack 8).
    cpp_pages = 30312
    group = _make_group(initial_mapped_chunks=1895, max_chunks=1895)

    def fire(n_pages: int) -> None:
        nonlocal cpp_pages
        group.shrink(n_pages)  # arena side
        cpp_pages -= n_pages  # C++ side caps exactly n_pages
        assert _physical_pages(group) >= cpp_pages, (
            f"arena unmapped below C++ boundary: physical="
            f"{_physical_pages(group)} < cpp={cpp_pages}"
        )

    for _ in range(16):
        fire(64)  # 4 chunks each, perfectly aligned
    # These three 8-page fires are what triggered the crash under the old
    # ceil-per-call behavior (3 chunks unmapped for 24 pages released).
    for _ in range(3):
        fire(8)

    # Memory is still genuinely reclaimed, not leaked: total unmapped chunks
    # equals floor(total_released_bytes / CHUNK), with < 1 chunk carried.
    total_released_pages = 16 * 64 + 3 * 8
    total_released_bytes = total_released_pages * group._bytes_per_page_per_layer
    expected_unmapped_chunks = total_released_bytes // CHUNK_SIZE_BYTES
    assert group.mapped_chunks == 1895 - expected_unmapped_chunks
    assert 0 <= group._shrink_debt_bytes < CHUNK_SIZE_BYTES
    # Final C++ size matches the production trace (29264 pages).
    assert cpp_pages == 29264


def test_new_behavior_strictly_safer_than_ceil_per_call() -> None:
    """Documents the fix: a pure ceil-per-call would unmap strictly more."""
    group = _make_group(initial_mapped_chunks=63, max_chunks=63)
    ceil_per_call_chunks = 0
    for n_pages in [8, 8, 8]:
        group.shrink(n_pages)
        ceil_per_call_chunks += max(
            1, math.ceil(n_pages * group._bytes_per_page_per_layer / CHUNK_SIZE_BYTES)
        )
    actual_unmapped = 63 - group.mapped_chunks
    # ceil-per-call would have unmapped 3 chunks; the fix unmaps only 1.
    assert ceil_per_call_chunks == 3
    assert actual_unmapped == 1


def test_shrink_then_grow_reuses_carry_without_remapping() -> None:
    """A sub-chunk shrink followed by a grow of the same size is a no-op
    physically (the still-mapped carry backs the grow), and never exceeds
    max_chunks."""
    group = _make_group(initial_mapped_chunks=10, max_chunks=10)
    start = group.mapped_chunks
    group.shrink(8)  # < 1 chunk -> carried, 0 chunks unmapped
    assert group.mapped_chunks == start
    group.grow(8)  # covered entirely by carry -> 0 chunks mapped
    assert group.mapped_chunks == start
    assert group._shrink_debt_bytes == 0


def test_grow_maps_only_the_shortfall_beyond_carry() -> None:
    """Grow larger than the carried slack maps exactly the ceil of the
    shortfall and updates the carry."""
    group = _make_group(initial_mapped_chunks=5, max_chunks=20)
    group.shrink(8)  # carry = 8 pages worth (half a chunk), 0 chunks unmapped
    assert group.mapped_chunks == 5
    group.grow(24)  # need 24 pages; 8 covered by carry -> shortfall 16 = 1 chunk
    assert group.mapped_chunks == 6
    assert group._shrink_debt_bytes == 0
