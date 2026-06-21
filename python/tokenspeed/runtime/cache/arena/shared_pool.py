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

"""Shared physical handle pool for CUDA VMM arena."""

from __future__ import annotations

CHUNK_SIZE_BYTES = 2 * 1024 * 1024


class SharedHandlePool:
    """Pool of physical allocation handles keyed by chunk index.

    Two modes:

    * ``device is None`` (default) -- logical stub mode.  ``get_handle`` returns
      a deterministic placeholder integer.  Used by CPU-only unit tests and any
      environment without a GPU; no real device memory is allocated.
    * ``device`` set to a CUDA device ordinal -- real mode.  Each chunk lazily
      allocates ``chunk_size`` bytes via ``cuMemCreate`` and the returned
      ``CUmemGenericAllocationHandle`` is cached and reused.  Call
      :meth:`release_all` to free the physical pages.
    """

    def __init__(
        self,
        num_chunks: int,
        *,
        device: int | None = None,
        chunk_size: int = CHUNK_SIZE_BYTES,
    ) -> None:
        if num_chunks <= 0:
            raise ValueError("num_chunks must be positive")
        self.num_chunks = num_chunks
        self.device = device
        self.chunk_size = chunk_size
        self._handles: dict[int, int] = {}

    def get_handle(self, chunk_id: int) -> int:
        if chunk_id < 0 or chunk_id >= self.num_chunks:
            raise IndexError(f"chunk_id {chunk_id} out of range [0, {self.num_chunks})")
        if chunk_id not in self._handles:
            if self.device is None:
                # Logical stub handle (deterministic, never zero).
                self._handles[chunk_id] = chunk_id + 1
            else:
                from tokenspeed.runtime.cache.arena._cuda_vmm import cu_mem_create

                self._handles[chunk_id] = cu_mem_create(self.chunk_size, self.device)
        return self._handles[chunk_id]

    def release_all(self) -> None:
        """Free every physical allocation held by this pool."""
        if self.device is not None:
            from tokenspeed.runtime.cache.arena._cuda_vmm import cu_mem_release

            for handle in self._handles.values():
                cu_mem_release(handle)
        self._handles.clear()
