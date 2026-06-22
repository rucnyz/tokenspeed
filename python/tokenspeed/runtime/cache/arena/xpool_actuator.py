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

"""Python-side XPool fire actuator executed on a background thread."""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tokenspeed.runtime.cache.arena.chunk_arena import ChunkArena
    from tokenspeed.runtime.cache.arena.kv_arena import KvLayerArenaGroup

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FirePlan:
    direction: str
    page_ids: list[int]
    op_id: int = 0
    # Raw C++ plan object kept so we can call scheduler.apply_xpool_fire after
    # the VMM ops complete (None for unit-test / stub plans).
    cpp_plan: Any = field(default=None, repr=False)


class XPoolActuator:
    """Execute cuMemUnmap/Map transfers off the decode critical path.

    After the physical memory remapping completes, ``apply_xpool_fire`` is
    called on the C++ scheduler to update KV and mamba allocator capacities and
    clear the pending fire latch.  This closes the loop between the budgeter
    (which generates the plan) and the allocators (which act on capacity).
    """

    #: Maximum seconds to wait for in-flight capped pages to drain before
    #: proceeding with the physical unmap.  A warning is emitted on timeout.
    DRAIN_TIMEOUT_S: float = 30.0
    #: Poll interval (seconds) while waiting for drain to complete.
    DRAIN_POLL_S: float = 0.005

    def __init__(
        self,
        *,
        kv_arena: ChunkArena | KvLayerArenaGroup,
        mamba_arena: ChunkArena,
        scheduler: Any | None = None,
        kv_bytes_per_page: int = 0,
    ) -> None:
        self.kv_arena = kv_arena
        self.mamba_arena = mamba_arena
        # Optional: if provided, apply_xpool_fire is called after each
        # successful VMM operation to update C++ allocator capacities.
        self._scheduler = scheduler
        # kv_bytes_per_page is used to convert KV page counts to mamba chunk
        # counts when the two arenas have different byte-per-unit ratios.
        self._kv_bytes_per_page = kv_bytes_per_page
        self._lock = threading.Lock()
        self._inflight = False
        # The C++ budgeter latches its pending plan (op_id starts at 1), so we
        # de-dup here by the last actuated op_id; 0 means "nothing yet".
        self._last_op_id = 0

    def maybe_execute(self, plan: object) -> bool:
        """Actuate a budgeter plan unless it was already actuated.

        Args:
            plan: any object exposing ``op_id`` (int), ``direction`` (str) and
                ``page_ids`` (iterable of int) -- e.g. the C++ ``XPoolFirePlan``.

        Returns:
            True if a new transfer was launched, False if the plan was a
            duplicate of the previously actuated one.
        """
        op_id = int(plan.op_id)
        if op_id == self._last_op_id:
            return False
        self._last_op_id = op_id
        fire_plan = FirePlan(
            direction=str(plan.direction),
            page_ids=list(plan.page_ids),
            op_id=op_id,
            cpp_plan=plan,
        )
        logger.info(
            "XPool fire dispatched: op_id=%d direction=%s n_pages=%d",
            op_id,
            fire_plan.direction,
            len(fire_plan.page_ids),
        )
        self.execute_async(fire_plan)
        return True

    def execute_async(self, plan: FirePlan) -> None:
        worker = threading.Thread(
            target=self._execute_locked, args=(plan,), daemon=True
        )
        worker.start()

    def _execute_locked(self, plan: FirePlan) -> None:
        with self._lock:
            if self._inflight:
                raise RuntimeError("XPoolActuator: concurrent fire is not supported")
            self._inflight = True
            try:
                vmm_done = self._do_vmm(plan)
                if self._scheduler is not None and plan.cpp_plan is not None:
                    if vmm_done:
                        # Physical transfer completed: update C++ allocator
                        # capacities and clear the pending fire latch.
                        try:
                            self._scheduler.apply_xpool_fire(plan.cpp_plan)
                            logger.info(
                                "XPool fire committed: op_id=%d direction=%s",
                                plan.op_id,
                                plan.direction,
                            )
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "apply_xpool_fire failed (op_id=%s dir=%s): %s",
                                plan.op_id,
                                plan.direction,
                                exc,
                            )
                    else:
                        # VMM was skipped (headroom exhausted or other guard).
                        # Clear latch only so the budgeter can emit new plans,
                        # but do NOT update allocator capacities.
                        cancel_fn = getattr(
                            self._scheduler, "cancel_xpool_fire", None
                        )
                        if cancel_fn is not None:
                            try:
                                cancel_fn()
                            except Exception as exc:  # noqa: BLE001
                                logger.warning(
                                    "cancel_xpool_fire failed (op_id=%s): %s",
                                    plan.op_id,
                                    exc,
                                )
                        logger.info(
                            "XPool fire cancelled (VMM skipped): "
                            "op_id=%d direction=%s",
                            plan.op_id,
                            plan.direction,
                        )
            except Exception as exc:  # noqa: BLE001
                # Background threads silently swallow uncaught exceptions.
                # Log explicitly so failures are always visible.
                logger.warning(
                    "XPool fire failed: op_id=%d direction=%s error=%s",
                    plan.op_id,
                    plan.direction,
                    exc,
                )
            finally:
                self._inflight = False

    def _kv_pages_to_mamba_chunks(self, n_kv_pages: int) -> int:
        """Convert a KV page count to the corresponding mamba chunk count.

        Both sides of a transfer represent the same byte volume.  The mamba
        arena uses raw chunk counts (2 MB each), so we round up from the KV
        byte total.
        """
        if self._kv_bytes_per_page <= 0:
            return n_kv_pages  # fallback: treat 1 page ≈ 1 chunk
        from tokenspeed.runtime.cache.arena._cuda_vmm import CHUNK_SIZE_BYTES

        return max(
            1, math.ceil(n_kv_pages * self._kv_bytes_per_page / CHUNK_SIZE_BYTES)
        )

    def _wait_drain(self, drain_fn_name: str = "has_capped_kv_inflight") -> None:
        """Poll until no capped pages/slots remain in-flight (or timeout).

        Args:
            drain_fn_name: Name of the C++ scheduler method to poll.  Defaults
                to ``has_capped_kv_inflight`` for kv_to_mamba direction.  Pass
                ``has_capped_mamba_inflight`` for mamba_to_kv direction.
        """
        if self._scheduler is None:
            return
        drain_fn = getattr(self._scheduler, drain_fn_name, None)
        if drain_fn is None:
            return
        deadline = time.monotonic() + self.DRAIN_TIMEOUT_S
        while drain_fn():
            if time.monotonic() > deadline:
                logger.warning(
                    "XPool drain timeout after %.1f s (fn=%s); proceeding with "
                    "unmap (some in-flight pages may still be in use)",
                    self.DRAIN_TIMEOUT_S,
                    drain_fn_name,
                )
                break
            time.sleep(self.DRAIN_POLL_S)

    # ------------------------------------------------------------------
    # Helper: decide whether the kv_arena supports physical handle transfer
    # (i.e. it is a KvLayerArenaGroup with shrink_with_handles).
    # ------------------------------------------------------------------

    def _kv_supports_handle_transfer(self) -> bool:
        return callable(getattr(self.kv_arena, "shrink_with_handles", None))

    def _balance_handles(
        self,
        raw_handles: list[int],
        n_needed: int,
    ) -> list[int]:
        """Return exactly *n_needed* physical handles.

        * If ``raw_handles`` has more than needed the excess are freed via
          ``cuMemRelease`` (no waste).
        * If fewer than needed, new handles are allocated via ``cuMemCreate``
          on the first available device.
        """
        from tokenspeed.runtime.cache.arena._cuda_vmm import CHUNK_SIZE_BYTES

        diff = len(raw_handles) - n_needed
        if diff > 0:
            # Release excess handles.
            from tokenspeed.runtime.cache.arena._cuda_vmm import cu_mem_release

            for h in raw_handles[n_needed:]:
                try:
                    cu_mem_release(h)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("_balance_handles: cuMemRelease failed: %s", exc)
            logger.debug(
                "_balance_handles: released %d excess KV handles", diff
            )
            return raw_handles[:n_needed]

        elif diff < 0:
            # Allocate extra handles to make up the shortfall.
            from tokenspeed.runtime.cache.arena._cuda_vmm import cu_mem_create

            device = getattr(self.mamba_arena, "_device", None)
            if device is None:
                device = getattr(self.kv_arena, "_device", None)
            extra: list[int] = []
            for _ in range(-diff):
                extra.append(cu_mem_create(CHUNK_SIZE_BYTES, device))
            logger.debug(
                "_balance_handles: allocated %d extra handles", -diff
            )
            return raw_handles + extra

        return raw_handles  # exact match

    def _do_vmm(self, plan: FirePlan) -> bool:
        """Perform physical cuMemUnmap / cuMemMap operations.

        When the arenas support physical handle transfer (pre-construction
        path with :class:`KvLayerArenaGroup`), the KV handles are donated to
        the mamba arena rather than released and re-allocated.  This means
        **no net change in total GPU physical memory** per fire.

        For ``kv_to_mamba`` the KV pool is capped and drained before unmap.
        For ``mamba_to_kv`` no drain is needed (we are adding KV capacity).

        Returns:
            True if the physical VMM transfer was executed, False if skipped
            (e.g. arena headroom exhausted).  Caller should call
            ``apply_xpool_fire`` only when True, and ``cancel_xpool_fire`` when
            False.
        """
        n_kv_pages = len(plan.page_ids)
        n_mamba_chunks = self._kv_pages_to_mamba_chunks(n_kv_pages)
        use_transfer = self._kv_supports_handle_transfer()

        if plan.direction == "mamba_to_kv":
            # KvLayerArenaGroup exposes headroom_pages (in KV-page units) which
            # already accounts for per-layer chunk → page conversion.  Fall back
            # to the raw max_chunks/mapped_chunks difference for a plain
            # ChunkArena (where 1 chunk ≈ 1 page in the caller's mapping).
            if hasattr(self.kv_arena, "headroom_pages"):
                kv_available = self.kv_arena.headroom_pages
            else:
                kv_available = getattr(self.kv_arena, "max_chunks", 0) - getattr(
                    self.kv_arena, "mapped_chunks", 0
                )
            if kv_available < n_kv_pages:
                logger.warning(
                    "mamba_to_kv fire skipped (op_id=%d): kv arena headroom "
                    "exhausted (headroom=%d pages, need %d pages)",
                    plan.op_id,
                    kv_available,
                    n_kv_pages,
                )
                return False

            # Shrink-and-drain: cap the tail mamba slots in C++ BEFORE unmap so
            # no new allocations land on slots that are about to be transferred.
            if self._scheduler is not None:
                prepare_fn = getattr(
                    self._scheduler, "prepare_mamba_to_kv_fire", None
                )
                if prepare_fn is not None:
                    try:
                        prepare_fn(n_mamba_chunks)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "prepare_mamba_to_kv_fire failed (n=%d): %s",
                            n_mamba_chunks, exc,
                        )
                        return False
            self._wait_drain("has_capped_mamba_inflight")

            if use_transfer:
                # True handle transfer: shrink mamba → donate handles → grow KV.
                raw = self.mamba_arena.shrink_with_handles(n_mamba_chunks)
                balanced = self._balance_handles(raw, n_kv_pages)
                self.kv_arena.grow_with_handles(balanced, n_kv_pages)
                logger.info(
                    "mamba_to_kv VMM done (handle transfer): "
                    "shrunk %d mamba chunks, grew %d kv pages "
                    "(%d handles transferred)",
                    n_mamba_chunks, n_kv_pages, len(balanced),
                )
            else:
                self.mamba_arena.shrink(n_mamba_chunks)
                self.kv_arena.grow(n_kv_pages)
                logger.info(
                    "mamba_to_kv VMM done: shrunk %d mamba chunks, grew %d kv pages",
                    n_mamba_chunks, n_kv_pages,
                )
            return True

        elif plan.direction == "kv_to_mamba":
            mamba_available = getattr(self.mamba_arena, "max_chunks", 0) - getattr(
                self.mamba_arena, "mapped_chunks", 0
            )
            if mamba_available < n_mamba_chunks:
                logger.warning(
                    "kv_to_mamba fire skipped (op_id=%d): mamba arena headroom "
                    "exhausted (%d/%d mapped, need %d more chunks)",
                    plan.op_id,
                    getattr(self.mamba_arena, "mapped_chunks", "?"),
                    getattr(self.mamba_arena, "max_chunks", "?"),
                    n_mamba_chunks,
                )
                return False

            if self._scheduler is not None:
                prepare_fn = getattr(self._scheduler, "prepare_kv_to_mamba_fire", None)
                if prepare_fn is not None:
                    try:
                        prepare_fn(n_kv_pages)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "prepare_kv_to_mamba_fire failed (n=%d): %s",
                            n_kv_pages, exc,
                        )
                        return False
            self._wait_drain()

            if use_transfer:
                # True handle transfer: shrink KV → donate handles → grow mamba.
                raw = self.kv_arena.shrink_with_handles(n_kv_pages)
                balanced = self._balance_handles(raw, n_mamba_chunks)
                self.mamba_arena.grow_with_handles(balanced)
                logger.info(
                    "kv_to_mamba VMM done (handle transfer): "
                    "shrunk %d kv pages, grew %d mamba chunks "
                    "(%d handles transferred)",
                    n_kv_pages, n_mamba_chunks, len(balanced),
                )
            else:
                self.kv_arena.shrink(n_kv_pages)
                self.mamba_arena.grow(n_mamba_chunks)
                logger.info(
                    "kv_to_mamba VMM done: shrunk %d kv pages, grew %d mamba chunks",
                    n_kv_pages, n_mamba_chunks,
                )
            return True

        else:
            raise ValueError(f"unknown fire direction: {plan.direction}")
