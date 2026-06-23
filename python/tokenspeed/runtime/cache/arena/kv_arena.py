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

"""Per-layer KV pool VMM arena group for HiMA cross-pool transfers."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenspeed.runtime.cache.arena.chunk_arena import ChunkArena

from tokenspeed.runtime.cache.arena._cuda_vmm import CHUNK_SIZE_BYTES

logger = logging.getLogger(__name__)


class KvLayerArenaGroup:
    """Manages 2 × layer_num :class:`ChunkArena` objects (one per k/v buffer).

    Each layer's k-buffer and v-buffer gets its own dedicated VA window so that
    the per-layer ``(total_tokens, head_num, head_dim)`` layout used by
    attention kernels remains contiguous while still being VMM-backed.

    :meth:`grow` / :meth:`shrink` accept a *number of KV pages* and
    transparently convert to the per-layer chunk count, so callers share the
    same ``n_kv_pages`` unit as the C++ allocator.

    Args:
        k_arenas: One :class:`ChunkArena` per layer for K buffers.
        v_arenas: One :class:`ChunkArena` per layer for V buffers.
        page_size: Token slots per KV page (= ``block_size``).
        head_num: Number of KV heads per TP rank.
        head_dim: Head dimension.
        dtype_itemsize: Bytes per element (``store_dtype.itemsize``).
        initial_live_rows: Token-row count that is initially physically mapped.
            Used by :meth:`~MHATokenToKVPool.bind_arena` to know how many rows
            to zero-initialise.
    """

    def __init__(
        self,
        *,
        k_arenas: list[ChunkArena],
        v_arenas: list[ChunkArena],
        page_size: int,
        head_num: int,
        head_dim: int,
        dtype_itemsize: int,
        initial_live_rows: int = 0,
    ) -> None:
        if len(k_arenas) != len(v_arenas):
            raise ValueError("k_arenas and v_arenas must have equal length")
        self._k_arenas = k_arenas
        self._v_arenas = v_arenas
        self._layer_num = len(k_arenas)
        self._page_size = page_size
        # Per-layer bytes added/removed for one KV page (= page_size rows).
        self._bytes_per_page_per_layer = (
            page_size * head_num * head_dim * dtype_itemsize
        )
        self.initial_live_rows = initial_live_rows

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def layer_num(self) -> int:
        return self._layer_num

    def k_base_ptr(self, layer_idx: int) -> int:
        """VA base pointer for k_buffer[layer_idx]."""
        return self._k_arenas[layer_idx].base_ptr

    def v_base_ptr(self, layer_idx: int) -> int:
        """VA base pointer for v_buffer[layer_idx]."""
        return self._v_arenas[layer_idx].base_ptr

    # ------------------------------------------------------------------
    # VMM grow / shrink
    # ------------------------------------------------------------------

    def _pages_to_chunks(self, n_pages: int) -> int:
        """Convert a KV page count to a per-layer chunk count (rounds up)."""
        return max(
            1, math.ceil(n_pages * self._bytes_per_page_per_layer / CHUNK_SIZE_BYTES)
        )

    def grow(self, n_kv_pages: int) -> None:
        """Map ``n_kv_pages`` worth of additional per-layer physical chunks.

        Called after a ``mamba_to_kv`` fire: physical pages are added to the
        tail of every per-layer k/v arena so that the C++ allocator can
        distribute them to new requests.
        """
        if n_kv_pages <= 0:
            return
        n_chunks = self._pages_to_chunks(n_kv_pages)
        errors: list[str] = []
        for arena in self._k_arenas + self._v_arenas:
            try:
                arena.grow(n_chunks)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{arena.name}: {exc}")
        if errors:
            raise RuntimeError(
                f"KvLayerArenaGroup.grow failed on {len(errors)} arenas: "
                + "; ".join(errors)
            )
        logger.debug(
            "KvLayerArenaGroup grew %d pages (%d chunks/layer × %d arenas)",
            n_kv_pages,
            n_chunks,
            2 * self._layer_num,
        )

    def shrink(self, n_kv_pages: int) -> None:
        """Unmap ``n_kv_pages`` worth of tail per-layer physical chunks.

        Called before a ``kv_to_mamba`` fire (after drain completes): physical
        backing is removed from the tail of every per-layer k/v arena.
        Physical handles are kept in each sub-arena's ``shared_pool``; use
        :meth:`shrink_with_handles` to additionally extract the raw handles
        for cross-pool donation.
        """
        if n_kv_pages <= 0:
            return
        n_chunks = self._pages_to_chunks(n_kv_pages)
        errors: list[str] = []
        for arena in self._k_arenas + self._v_arenas:
            try:
                arena.shrink(n_chunks)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{arena.name}: {exc}")
        if errors:
            raise RuntimeError(
                f"KvLayerArenaGroup.shrink failed on {len(errors)} arenas: "
                + "; ".join(errors)
            )
        logger.debug(
            "KvLayerArenaGroup shrank %d pages (%d chunks/layer × %d arenas)",
            n_kv_pages,
            n_chunks,
            2 * self._layer_num,
        )

    def shrink_with_handles(self, n_kv_pages: int) -> list[int]:
        """Unmap per-layer tail chunks and return all freed raw handles.

        Combines :meth:`shrink` with handle extraction from every sub-arena's
        ``shared_pool``.  The returned list of raw
        ``CUmemGenericAllocationHandle`` values can be passed directly to
        :meth:`~tokenspeed.runtime.cache.arena.chunk_arena.ChunkArena.grow_with_handles`
        on the destination arena for a zero-allocation physical transfer.

        The total number of handles returned is
        ``2 × layer_num × n_chunks_per_layer`` which may differ from the
        number of mamba chunks needed (see :meth:`XPoolActuator._do_vmm_with_transfer`
        for the balancing logic).
        """
        if n_kv_pages <= 0:
            return []
        n_chunks = self._pages_to_chunks(n_kv_pages)
        all_handles: list[int] = []
        errors: list[str] = []
        for arena in self._k_arenas + self._v_arenas:
            try:
                handles = arena.shrink_with_handles(n_chunks)
                all_handles.extend(handles)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{arena.name}: {exc}")
        if errors:
            raise RuntimeError(
                f"KvLayerArenaGroup.shrink_with_handles failed: " + "; ".join(errors)
            )
        logger.debug(
            "KvLayerArenaGroup shrink_with_handles: %d pages → "
            "%d chunks/layer × %d arenas = %d handles extracted",
            n_kv_pages,
            n_chunks,
            2 * self._layer_num,
            len(all_handles),
        )
        return all_handles

    def grow_with_handles(self, raw_handles: list[int], n_kv_pages: int) -> None:
        """Map donated physical handles across all per-layer k/v arenas.

        Distributes *raw_handles* evenly among the ``2 × layer_num`` sub-arenas
        so each layer grows by the same number of chunks.  This is the
        reverse of :meth:`shrink_with_handles` and is used for
        ``mamba_to_kv`` transfers.

        Args:
            raw_handles: raw ``CUmemGenericAllocationHandle`` values donated
                by the mamba arena.
            n_kv_pages: target page count (used for logging only).
        """
        if not raw_handles:
            return
        n_arenas = 2 * self._layer_num
        n_each, remainder = divmod(len(raw_handles), n_arenas)
        if n_each == 0:
            raise RuntimeError(
                f"KvLayerArenaGroup.grow_with_handles: only {len(raw_handles)} "
                f"handles for {n_arenas} arenas (need ≥ {n_arenas})"
            )
        errors: list[str] = []
        arenas = self._k_arenas + self._v_arenas
        offset = 0
        for i, arena in enumerate(arenas):
            # Distribute remainder to the first *remainder* arenas.
            count = n_each + (1 if i < remainder else 0)
            try:
                arena.grow_with_handles(raw_handles[offset : offset + count])
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{arena.name}: {exc}")
            offset += count
        if errors:
            raise RuntimeError(
                f"KvLayerArenaGroup.grow_with_handles failed: " + "; ".join(errors)
            )
        logger.debug(
            "KvLayerArenaGroup grow_with_handles: %d handles → "
            "%d pages across %d arenas",
            len(raw_handles),
            n_kv_pages,
            n_arenas,
        )

    # ------------------------------------------------------------------
    # Headroom helpers (used by XPoolActuator for mamba_to_kv checks)
    # ------------------------------------------------------------------

    @property
    def max_chunks(self) -> int:
        """Per-layer maximum VMM chunks (baseline + mamba_to_kv headroom).

        Exposed so :class:`~tokenspeed.runtime.cache.arena.xpool_actuator.XPoolActuator`
        can check available KV headroom using the same attribute name as
        :class:`~tokenspeed.runtime.cache.arena.chunk_arena.ChunkArena`.
        Returns the value from the first K-arena (all layers share the same
        window size).
        """
        return self._k_arenas[0].max_chunks if self._k_arenas else 0

    @property
    def mapped_chunks(self) -> int:
        """Per-layer currently mapped VMM chunks.

        See :attr:`max_chunks` for usage context.
        """
        return self._k_arenas[0].mapped_chunks if self._k_arenas else 0

    @property
    def headroom_pages(self) -> int:
        """Number of additional KV *pages* this group can grow by.

        Converts the per-layer VMM chunk headroom
        (``max_chunks - mapped_chunks``) back into KV page units so that
        :class:`~tokenspeed.runtime.cache.arena.xpool_actuator.XPoolActuator`
        can compare directly against ``n_kv_pages`` without needing to know
        the per-layer chunk conversion factor.

        Returns 0 when the arenas are not initialised or the per-layer byte
        sizes are unknown.
        """
        if not self._k_arenas or self._bytes_per_page_per_layer <= 0:
            return 0
        headroom_chunks = self._k_arenas[0].max_chunks - self._k_arenas[0].mapped_chunks
        if headroom_chunks <= 0:
            return 0
        return int(headroom_chunks * CHUNK_SIZE_BYTES // self._bytes_per_page_per_layer)

    def close(self) -> None:
        """Release all per-layer arena VA windows."""
        for arena in self._k_arenas + self._v_arenas:
            try:
                arena.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "KvLayerArenaGroup.close error on %s: %s", arena.name, exc
                )
