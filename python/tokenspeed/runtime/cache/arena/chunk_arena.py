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

"""VA window with per-chunk map/unmap for a single logical pool."""

from __future__ import annotations

from tokenspeed.runtime.cache.arena._cuda_vmm import (
    CHUNK_SIZE_BYTES,
    cu_mem_address_free,
    cu_mem_address_reserve,
    cu_mem_map,
    cu_mem_set_access,
    cu_mem_unmap,
)
from tokenspeed.runtime.cache.arena.shared_pool import SharedHandlePool


class ChunkArena:
    """Reserve a VA window and map physical handles on demand.

    When ``shared_pool`` is backed by a real CUDA device (``device`` set), each
    mapped chunk additionally receives read/write access via ``cuMemSetAccess``
    so device kernels can touch the memory.  In stub mode (no device) the access
    grant is skipped.
    """

    def __init__(
        self,
        *,
        shared_pool: SharedHandlePool,
        max_chunks: int,
        mapped_chunks: int,
        name: str,
    ) -> None:
        if mapped_chunks < 0 or mapped_chunks > max_chunks:
            raise ValueError("mapped_chunks must be within [0, max_chunks]")
        self.shared_pool = shared_pool
        self.max_chunks = max_chunks
        self.mapped_chunks = mapped_chunks
        self.name = name
        self.chunk_size = shared_pool.chunk_size
        self._device = shared_pool.device
        self._base_ptr = cu_mem_address_reserve(max_chunks * self.chunk_size)
        self._mapped: set[int] = set()
        for chunk_id in range(mapped_chunks):
            self._map_chunk(chunk_id)

    @property
    def base_ptr(self) -> int:
        return self._base_ptr

    def _slot_ptr(self, chunk_id: int) -> int:
        return self._base_ptr + chunk_id * self.chunk_size

    def _map_chunk(self, chunk_id: int) -> int:
        if chunk_id in self._mapped:
            return chunk_id
        handle = self.shared_pool.get_handle(chunk_id)
        ptr = self._slot_ptr(chunk_id)
        cu_mem_map(ptr, self.chunk_size, handle)
        if self._device is not None:
            cu_mem_set_access(ptr, self.chunk_size, self._device)
        self._mapped.add(chunk_id)
        return chunk_id

    def grow(self, n_chunks: int) -> list[int]:
        grown: list[int] = []
        for _ in range(n_chunks):
            if self.mapped_chunks >= self.max_chunks:
                raise RuntimeError(f"{self.name}: grow exceeds reserved VA")
            chunk_id = self._map_chunk(self.mapped_chunks)
            grown.append(chunk_id)
            self.mapped_chunks += 1
        return grown

    def shrink(self, n_chunks: int) -> list[int]:
        """Unmap tail chunks and return their chunk IDs.

        The underlying physical handles are **kept** in ``shared_pool`` and
        are NOT freed.  Callers that want to discard the physical memory
        should call ``shared_pool.extract_handle`` on the returned IDs and
        then call ``cu_mem_release``; callers that want to transfer the
        physical memory to another arena should call
        :meth:`shrink_with_handles` instead.
        """
        released: list[int] = []
        for _ in range(n_chunks):
            if self.mapped_chunks <= 0:
                break
            chunk_id = self.mapped_chunks - 1
            if chunk_id in self._mapped:
                cu_mem_unmap(self._slot_ptr(chunk_id), self.chunk_size)
                self._mapped.remove(chunk_id)
            released.append(chunk_id)
            self.mapped_chunks -= 1
        return released

    def shrink_with_handles(self, n_chunks: int) -> list[int]:
        """Unmap tail chunks and return the raw ``CUmemGenericAllocationHandle``
        values for each freed chunk.

        Unlike :meth:`shrink`, the physical handles are *removed* from
        ``shared_pool`` so the caller takes ownership.  Pass the returned
        list to :meth:`grow_with_handles` on another arena to perform a
        zero-allocation physical-memory transfer.
        """
        chunk_ids = self.shrink(n_chunks)
        raw_handles: list[int] = []
        for cid in chunk_ids:
            try:
                raw_handles.append(self.shared_pool.extract_handle(cid))
            except KeyError:
                pass  # chunk was never physically allocated (stub / already extracted)
        return raw_handles

    def grow_with_handles(self, raw_handles: list[int]) -> None:
        """Map pre-existing physical handles to the tail of this arena's VA.

        Each handle in *raw_handles* is injected into ``shared_pool`` at the
        next available chunk slot (== current ``mapped_chunks``) and then
        immediately mapped to the corresponding VA position.  No
        ``cuMemCreate`` is called — the physical pages come from the caller.

        Use together with :meth:`shrink_with_handles` on a source arena to
        perform a zero-allocation inter-pool transfer (cuMemUnmap on source,
        cuMemMap on destination, one ``cuMemCreate`` saved per chunk).

        Raises ``RuntimeError`` if the injection would exceed ``max_chunks``.
        """
        for handle in raw_handles:
            if self.mapped_chunks >= self.max_chunks:
                raise RuntimeError(
                    f"{self.name}: grow_with_handles exceeds reserved VA "
                    f"(mapped={self.mapped_chunks}, max={self.max_chunks})"
                )
            cid = self.mapped_chunks
            # Pre-load the donated handle at this slot so _map_chunk (and
            # get_handle) will find it instead of calling cuMemCreate.
            self.shared_pool.inject_handle(cid, handle)
            self._map_chunk(cid)  # cu_mem_map + cu_mem_set_access
            self.mapped_chunks += 1

    def transfer_out(self, chunk_ids: list[int]) -> None:
        for chunk_id in chunk_ids:
            if chunk_id in self._mapped:
                cu_mem_unmap(self._slot_ptr(chunk_id), self.chunk_size)
                self._mapped.remove(chunk_id)

    def transfer_in(self, chunk_ids: list[int]) -> None:
        for chunk_id in chunk_ids:
            self._map_chunk(chunk_id)

    def close(self) -> None:
        """Unmap every chunk and free the reserved virtual address window."""
        for chunk_id in list(self._mapped):
            cu_mem_unmap(self._slot_ptr(chunk_id), self.chunk_size)
        self._mapped.clear()
        if self._base_ptr:
            cu_mem_address_free(self._base_ptr, self.max_chunks * self.chunk_size)
            self._base_ptr = 0
