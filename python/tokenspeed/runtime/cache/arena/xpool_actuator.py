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

# Keys in the per-fire timing breakdown dict returned by ``_do_vmm``.
FIRE_BREAKDOWN_PREPARE = "prepare_us"
FIRE_BREAKDOWN_DRAIN_POLL = "drain_poll_us"
FIRE_BREAKDOWN_DRAIN_SYNC = "drain_sync_us"
FIRE_BREAKDOWN_VMM = "vmm_us"


def _empty_fire_breakdown() -> dict[str, float]:
    return {
        FIRE_BREAKDOWN_PREPARE: 0.0,
        FIRE_BREAKDOWN_DRAIN_POLL: 0.0,
        FIRE_BREAKDOWN_DRAIN_SYNC: 0.0,
        FIRE_BREAKDOWN_VMM: 0.0,
    }


@dataclass(slots=True)
class FirePlan:
    direction: str
    page_ids: list[int]
    op_id: int = 0
    # Raw C++ plan object kept so we can call scheduler.apply_xpool_fire after
    # the VMM ops complete (None for unit-test / stub plans).
    cpp_plan: Any = field(default=None, repr=False)
    # S2.5-followup (early cap): set by maybe_execute() once the headroom
    # check + prepare_*_fire tail-cap has already run synchronously on the
    # main event-loop thread.  _do_vmm() must skip redoing that step and
    # reuse prepare_us (the elapsed time already measured) instead.
    already_prepared: bool = False
    prepare_us: float = 0.0


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
        # CUDA device index for the GPU this actuator operates on, used for
        # device-specific synchronization in _wait_drain.  Try mamba_arena
        # first (ChunkArena stores it as _device), then kv_arena, then fall
        # back to None (torch.cuda.synchronize(None) uses current device).
        self._device: int | None = (
            getattr(mamba_arena, "_device", None)
            or getattr(kv_arena, "_device", None)
            or getattr(getattr(kv_arena, "shared_pool", None), "device", None)
        )
        self._lock = threading.Lock()
        self._inflight = False
        # The C++ budgeter latches its pending plan (op_id starts at 1), so we
        # de-dup here by the last actuated op_id; 0 means "nothing yet".
        self._last_op_id = 0
        # Cumulative counters surfaced for replay / Prometheus-style metrics.
        # We bump them inside _execute_locked so a single fire can be reflected
        # in either `committed` (VMM succeeded -> apply_xpool_fire) or
        # `cancelled` (VMM skipped -> cancel_xpool_fire) but not both. Reads
        # are racy-but-stable: a metrics snapshot may witness +1 mid-fire,
        # which is fine — we only need the steady-state count.
        self.committed_kv_to_mamba: int = 0
        self.committed_mamba_to_kv: int = 0
        self.cancelled_kv_to_mamba: int = 0
        self.cancelled_mamba_to_kv: int = 0
        # S2.5: runtime EWMA of physical fire cost in microseconds per KV
        # page actually unmap/remapped.  Updated only on committed fires
        # (cancelled fires include no VMM work).  The EWMA half-life is
        # ~8 fires (alpha=0.25), enough to smooth jitter from short
        # transfers without hiding sustained drift.  Initial 0.0 means
        # "no samples yet"; calibrate_kappa.py treats that as missing.
        self.ewma_xfer_us_per_page: float = 0.0
        self.last_fire_us: float = 0.0
        self.last_fire_pages: int = 0
        # S2.5 follow-up: per-fire sub-stage timings (microseconds) from the
        # most recent committed fire.  Exposed via budget.jsonl for offline
        # breakdown analysis (see tools/calibrate_kappa.py).
        self.last_fire_prepare_us: float = 0.0
        self.last_fire_drain_poll_us: float = 0.0
        self.last_fire_drain_sync_us: float = 0.0
        self.last_fire_vmm_us: float = 0.0
        self._ewma_xfer_alpha: float = 0.25
        self._fires_observed: int = 0
        # S2.6: committed migration count (Stage-0: directed retraction).
        self.committed_migrate: int = 0
        self._last_migrate_op_id: int = 0

    def _record_fire_cost(self, n_pages: int, elapsed_us: float) -> None:
        """Update the runtime EWMA of microseconds per KV page transferred.

        Only called from the actuator background thread after a fire
        commits.  Cancelled fires don't update the EWMA because the VMM
        work was a no-op.  We normalise by the number of KV pages in the
        plan (the FirePlan.page_ids length) so the EWMA mirrors the
        scheduler's ``xpool_xfer_us_per_page`` cost parameter directly.
        """
        if n_pages <= 0 or elapsed_us <= 0.0:
            return
        per_page = elapsed_us / float(n_pages)
        self.last_fire_us = elapsed_us
        self.last_fire_pages = int(n_pages)
        self._fires_observed += 1
        if self.ewma_xfer_us_per_page <= 0.0:
            # First sample seeds the EWMA so it doesn't take many fires
            # to climb out of zero.  Subsequent samples blend in normally.
            self.ewma_xfer_us_per_page = per_page
        else:
            a = self._ewma_xfer_alpha
            self.ewma_xfer_us_per_page = (
                1.0 - a
            ) * self.ewma_xfer_us_per_page + a * per_page

    def _record_fire_breakdown(self, breakdown: dict[str, float]) -> None:
        """Store sub-stage timings from the most recent committed fire."""
        self.last_fire_prepare_us = float(breakdown.get(FIRE_BREAKDOWN_PREPARE, 0.0))
        self.last_fire_drain_poll_us = float(
            breakdown.get(FIRE_BREAKDOWN_DRAIN_POLL, 0.0)
        )
        self.last_fire_drain_sync_us = float(
            breakdown.get(FIRE_BREAKDOWN_DRAIN_SYNC, 0.0)
        )
        self.last_fire_vmm_us = float(breakdown.get(FIRE_BREAKDOWN_VMM, 0.0))

    def maybe_execute(self, plan: object) -> bool:
        """Actuate a budgeter plan unless it was already actuated.

        S2.5-followup (early cap): the CPU-only headroom check and
        ``prepare_*_fire`` tail-cap run synchronously here, on the calling
        (main event-loop) thread, *before* the background drain+transfer
        thread is even spawned.  Previously the tail-cap only took effect
        once the background thread got scheduled and reached ``_do_vmm``,
        which could be several event-loop iterations later under load —
        during that gap ``NextExecutionPlan()`` keeps admitting new
        requests, some of which land on the soon-to-be-unmapped tail pages
        and extend the drain wait.  Capping immediately (same iteration as
        the fire decision) closes that window: the very next
        ``NextExecutionPlan()`` call already sees the capped tail and stops
        admitting onto it, so ``_wait_drain()``'s later
        ``torch.cuda.synchronize()`` has fewer additional in-flight kernels
        to wait for.  This does not touch requests already using tail pages
        before the cap -- those still drain naturally.

        Args:
            plan: any object exposing ``op_id`` (int), ``direction`` (str) and
                ``page_ids`` (iterable of int) -- e.g. the C++ ``XPoolFirePlan``.

        Returns:
            True if a new transfer was launched or cancelled, False if the
            plan was a duplicate of the previously actuated one.
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

        prepare_us = self._check_and_prepare(fire_plan)
        if prepare_us is None:
            self._cancel_now(fire_plan)
            return True

        fire_plan.prepare_us = prepare_us
        fire_plan.already_prepared = True
        self.execute_async(fire_plan)
        return True

    def _cancel_now(self, plan: FirePlan) -> None:
        """Cancel a fire whose prepare step failed before any background
        thread or GPU work was ever dispatched (headroom exhausted or
        ``prepare_*_fire`` raised).  Mirrors the cancel branch in
        :meth:`_execute_locked` but runs synchronously on the caller's
        thread since no VMM/drain work happened.
        """
        if self._scheduler is not None:
            cancel_fn = getattr(self._scheduler, "cancel_xpool_fire", None)
            if cancel_fn is not None:
                try:
                    cancel_fn()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "cancel_xpool_fire failed (op_id=%s): %s", plan.op_id, exc
                    )
        logger.info(
            "XPool fire cancelled (prepare failed): op_id=%d direction=%s",
            plan.op_id,
            plan.direction,
        )
        if plan.direction == "kv_to_mamba":
            self.cancelled_kv_to_mamba += 1
        elif plan.direction == "mamba_to_kv":
            self.cancelled_mamba_to_kv += 1

    def _check_and_prepare(self, plan: FirePlan) -> float | None:
        """CPU-only headroom check + C++ tail-cap (no GPU sync involved).

        Safe to call synchronously from any thread: ``Shrink``/``SetCap`` on
        the C++ side only moves an allocator boundary, it does not touch GPU
        memory or wait on any kernel.

        Returns:
            Elapsed microseconds for the ``prepare_*_fire`` call (0.0 if
            there is no scheduler or no such method, e.g. unit-test stubs),
            or None if the fire must be cancelled: headroom is exhausted, or
            the scheduler's ``prepare_*_fire`` call raised.
        """
        n_kv_pages = len(plan.page_ids)
        n_mamba_chunks = self._kv_pages_to_mamba_chunks(n_kv_pages)
        mamba_static = self._mamba_is_static()

        if plan.direction == "mamba_to_kv":
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
                return None
            if self._scheduler is not None:
                prepare_fn = getattr(self._scheduler, "prepare_mamba_to_kv_fire", None)
                if prepare_fn is not None:
                    t0_ns = time.perf_counter_ns()
                    try:
                        prepare_fn(n_mamba_chunks)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "prepare_mamba_to_kv_fire failed (n=%d): %s",
                            n_mamba_chunks,
                            exc,
                        )
                        return None
                    return (time.perf_counter_ns() - t0_ns) / 1000.0
            return 0.0

        elif plan.direction == "kv_to_mamba":
            if not mamba_static:
                mamba_available = getattr(self.mamba_arena, "max_chunks", 0) - getattr(
                    self.mamba_arena, "mapped_chunks", 0
                )
                if mamba_available < n_mamba_chunks:
                    logger.warning(
                        "kv_to_mamba fire skipped (op_id=%d): mamba arena "
                        "headroom exhausted (%d/%d mapped, need %d more "
                        "chunks)",
                        plan.op_id,
                        getattr(self.mamba_arena, "mapped_chunks", "?"),
                        getattr(self.mamba_arena, "max_chunks", "?"),
                        n_mamba_chunks,
                    )
                    return None
            if self._scheduler is not None:
                prepare_fn = getattr(self._scheduler, "prepare_kv_to_mamba_fire", None)
                if prepare_fn is not None:
                    t0_ns = time.perf_counter_ns()
                    try:
                        prepare_fn(n_kv_pages)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "prepare_kv_to_mamba_fire failed (n=%d): %s",
                            n_kv_pages,
                            exc,
                        )
                        return None
                    return (time.perf_counter_ns() - t0_ns) / 1000.0
            return 0.0

        else:
            raise ValueError(f"unknown fire direction: {plan.direction}")

    def maybe_execute_migrate(self, plan: object) -> bool:
        """Stage-0 cross-pool migration: directed retraction of the best victim.

        When the admitter selects kCrossMigrate (both pools starved but active
        KV pages exist), we proactively retract the largest-footprint decoding
        request so its KV pages can be reclaimed for the incoming request.

        Stage-0 implementation: call ``scheduler.best_migrate_candidate()`` to
        find the victim, then let the scheduler's natural retraction path handle
        the KV write-back (existing L2 cache mechanism).  A future Stage-1
        will byte-copy the victim's KV tensors to host memory before retraction
        so the retracting request can reload without a full re-prefill.

        Args:
            plan: C++ ``XPoolMigratePlan`` object with ``op_id`` and
                ``pages_needed`` attributes.

        Returns:
            True if a migration was dispatched, False if this plan was already
            handled (duplicate op_id) or no suitable candidate exists.
        """
        op_id = int(plan.op_id)
        if op_id == self._last_migrate_op_id:
            return False
        self._last_migrate_op_id = op_id

        if self._scheduler is None:
            return False

        candidate_fn = getattr(self._scheduler, "best_migrate_candidate", None)
        apply_fn = getattr(self._scheduler, "apply_xpool_migrate", None)
        cancel_fn = getattr(self._scheduler, "cancel_xpool_migrate", None)

        if candidate_fn is None:
            if cancel_fn is not None:
                try:
                    cancel_fn()
                except Exception:  # noqa: BLE001
                    pass
            return False

        try:
            candidate_id: str = candidate_fn()
        except Exception as exc:  # noqa: BLE001
            logger.warning("best_migrate_candidate() failed: %s", exc)
            if cancel_fn is not None:
                try:
                    cancel_fn()
                except Exception:  # noqa: BLE001
                    pass
            return False

        if not candidate_id:
            logger.debug(
                "XPool migrate op_id=%d: no retraction candidate available "
                "(pages_needed=%d)",
                op_id,
                int(plan.pages_needed),
            )
            if cancel_fn is not None:
                try:
                    cancel_fn()
                except Exception:  # noqa: BLE001
                    pass
            return False

        # Stage-0: the scheduler's retraction path in NextExecutionPlan will
        # naturally pick this request as the next victim when KV pressure
        # peaks.  We commit the migration plan immediately so the budgeter can
        # issue the next plan; the physical KV writeback happens in the next
        # forward pass where the victim would have been retracted anyway.
        logger.info(
            "XPool migrate dispatched (stage-0 retraction): op_id=%d "
            "candidate=%s pages_needed=%d",
            op_id,
            candidate_id,
            int(plan.pages_needed),
        )
        if apply_fn is not None:
            try:
                apply_fn(plan)
                self.committed_migrate += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("apply_xpool_migrate failed (op_id=%d): %s", op_id, exc)
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
                # S2.5: time the VMM work so we can build a runtime EWMA of
                # the actuator's cost-per-page.  perf_counter_ns is monotonic
                # and immune to wall-clock jumps.
                t0_ns = time.perf_counter_ns()
                vmm_done, breakdown = self._do_vmm(plan)
                elapsed_us = (time.perf_counter_ns() - t0_ns) / 1000.0
                if self._scheduler is not None and plan.cpp_plan is not None:
                    if vmm_done:
                        try:
                            self._scheduler.apply_xpool_fire(plan.cpp_plan)
                            logger.info(
                                "XPool fire committed: op_id=%d direction=%s "
                                "elapsed_us=%.1f n_pages=%d "
                                "prepare_us=%.1f drain_poll_us=%.1f "
                                "drain_sync_us=%.1f vmm_us=%.1f",
                                plan.op_id,
                                plan.direction,
                                elapsed_us,
                                len(plan.page_ids),
                                breakdown[FIRE_BREAKDOWN_PREPARE],
                                breakdown[FIRE_BREAKDOWN_DRAIN_POLL],
                                breakdown[FIRE_BREAKDOWN_DRAIN_SYNC],
                                breakdown[FIRE_BREAKDOWN_VMM],
                            )
                            self._record_fire_cost(len(plan.page_ids), elapsed_us)
                            self._record_fire_breakdown(breakdown)
                            if plan.direction == "kv_to_mamba":
                                self.committed_kv_to_mamba += 1
                            elif plan.direction == "mamba_to_kv":
                                self.committed_mamba_to_kv += 1
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
                        cancel_fn = getattr(self._scheduler, "cancel_xpool_fire", None)
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
                        if plan.direction == "kv_to_mamba":
                            self.cancelled_kv_to_mamba += 1
                        elif plan.direction == "mamba_to_kv":
                            self.cancelled_mamba_to_kv += 1
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

    def _wait_drain(
        self, drain_fn_name: str = "has_capped_kv_inflight"
    ) -> tuple[float, float]:
        """Poll until no capped pages/slots remain in-flight, then GPU-sync.

        Two-phase drain:
        1. Poll the C++ scheduler until it reports no requests are using capped
           pages/slots (or timeout).  This ensures no NEW GPU kernels will be
           dispatched that touch those pages.
        2. Call ``torch.cuda.synchronize()`` to flush all in-flight GPU kernels
           that were already dispatched.  CUDA kernel launches are asynchronous:
           the C++ scheduler advances request state before the GPU finishes the
           corresponding decode/prefill kernel.  Without this synchronize,
           ``cu_mem_unmap`` can race with a running kernel that still reads the
           soon-to-be-unmapped KV-cache pages, producing
           ``CUDA error: an illegal memory access was encountered``.

        Args:
            drain_fn_name: Name of the C++ scheduler method to poll.  Defaults
                to ``has_capped_kv_inflight`` for kv_to_mamba direction.  Pass
                ``has_capped_mamba_inflight`` for mamba_to_kv direction.

        Returns:
            ``(poll_us, sync_us)`` — cumulative microseconds spent in the
            scheduler poll/sleep loop and in ``torch.cuda.synchronize()``
            calls respectively.
        """
        poll_us = 0.0
        sync_us = 0.0
        if self._scheduler is None:
            return poll_us, sync_us
        drain_fn = getattr(self._scheduler, drain_fn_name, None)
        if drain_fn is None:
            return poll_us, sync_us

        try:
            import torch.cuda  # noqa: PLC0415

            cuda_ok = torch.cuda.is_available()
        except Exception:  # noqa: BLE001
            cuda_ok = False

        def _sync() -> None:
            nonlocal sync_us
            if not cuda_ok:
                return
            t0_ns = time.perf_counter_ns()
            try:
                torch.cuda.synchronize(self._device)
            except Exception:  # noqa: BLE001
                pass
            sync_us += (time.perf_counter_ns() - t0_ns) / 1000.0

        deadline = time.monotonic() + self.DRAIN_TIMEOUT_S
        # Poll-then-sync-then-recheck: a request whose pages the C++ scheduler
        # just marked "drained" may still have its terminal decode/prefill
        # kernel sitting in an execution-stream queue that the overlap event
        # loop (event_loop_overlap) submitted *ahead* of committing that
        # request's results (that pipelining is the whole point of overlap
        # scheduling).  A single poll -> sync -> unmap sequence is vulnerable
        # to a request transitioning to "drained" in the tiny window between
        # our poll and our sync call.  We close that window by re-polling
        # after every sync and looping until a sync is immediately followed
        # by a clean (still-zero) drain check.
        while True:
            while drain_fn():
                if time.monotonic() > deadline:
                    logger.warning(
                        "XPool drain timeout after %.1f s (fn=%s); proceeding "
                        "with unmap (some in-flight pages may still be in use)",
                        self.DRAIN_TIMEOUT_S,
                        drain_fn_name,
                    )
                    _sync()
                    return poll_us, sync_us
                t_sleep_ns = time.perf_counter_ns()
                time.sleep(self.DRAIN_POLL_S)
                poll_us += (time.perf_counter_ns() - t_sleep_ns) / 1000.0
            # Phase 2: flush all pending GPU kernels on the correct device so
            # that no asynchronous CUDA work in flight at the moment the poll
            # above returned False can still touch the pages we're about to
            # unmap.  torch.cuda.synchronize(device) is equivalent to
            # cudaSetDevice(device) + cudaDeviceSynchronize() — it blocks
            # until every stream on that device (regardless of which CPU
            # thread submitted the work) is idle.
            _sync()
            # Re-check: if new capped-inflight usage appeared while we were
            # synchronizing, loop back to drain+sync again instead of racing
            # ahead with the unmap.
            if not drain_fn():
                return poll_us, sync_us
            if time.monotonic() > deadline:
                logger.warning(
                    "XPool drain timeout after %.1f s (fn=%s) on recheck; "
                    "proceeding with unmap",
                    self.DRAIN_TIMEOUT_S,
                    drain_fn_name,
                )
                return poll_us, sync_us

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
            logger.debug("_balance_handles: released %d excess KV handles", diff)
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
            logger.debug("_balance_handles: allocated %d extra handles", -diff)
            return raw_handles + extra

        return raw_handles  # exact match

    def _mamba_is_static(self) -> bool:
        """Detect a pre-mapped, fixed-size Mamba arena.

        Under the current SimpleMambaPool layout, ``conv_state`` and
        ``ssm_state`` occupy contiguous slices of the arena VA in the order
        ``[conv | ssm]``.  Growing or shrinking the tail of that VA window
        therefore moves the boundary of the SSM block, which the kernel
        addresses via tensor strides that were fixed at pool construction
        time — any physical unmap of those tail bytes will corrupt SSM
        state for slots the kernel still reads.

        HiMA Phase 3 (S2.2-followup) sizes the Python pool to
        ``base + xpool_mamba_headroom_slots`` slots at boot and the
        pre-arenas factory maps every chunk that backing requires.  The
        actuator then *skips* physical handle transfers on the Mamba side:
        ``apply_xpool_fire`` is still called so the C++ allocator's logical
        slot bound moves, but the Mamba arena's mapped_chunks never
        changes after boot.

        This method returns True for both the pre-construction path
        (factory creates a fully-pre-mapped ChunkArena where the headroom
        chunks ride along) and any future fully-pre-mapped variant.  When
        False the actuator falls back to the legacy handle-transfer path
        (still useful for unit tests with stub arenas).
        """
        max_chunks = getattr(self.mamba_arena, "max_chunks", None)
        mapped_chunks = getattr(self.mamba_arena, "mapped_chunks", None)
        if max_chunks is None or mapped_chunks is None:
            return False
        # Treat "fully mapped" or "arena holds extra VA but we never grow
        # into it" (the pre-construction factory case where the tensor
        # already covers the entire useful VA) as static.  We use equality
        # as the canonical signal; callers that want the old transfer path
        # can simply leave headroom_chunks > 0 in their arena.
        return int(max_chunks) == int(mapped_chunks)

    def _do_vmm(self, plan: FirePlan) -> tuple[bool, dict[str, float]]:
        """Perform physical cuMemUnmap / cuMemMap operations.

        The KV pool always uses physical mapping (handle release/re-map
        through its own per-layer arenas).  The Mamba side is *logical
        only* when the arena is pre-mapped at its full extent (see
        :meth:`_mamba_is_static`), because the conv/ssm contiguous-slice
        layout cannot tolerate post-boot resize.

        For ``kv_to_mamba`` the KV pool is capped and drained before
        unmap.  For ``mamba_to_kv`` we drain Mamba's capped tail then
        re-map physical pages in the KV arena.

        Returns:
            ``(committed, breakdown)`` where *committed* is True if the fire
            should be committed (``apply_xpool_fire`` called), False if it
            must be cancelled, and *breakdown* holds per-stage microsecond
            timings (``prepare_us``, ``drain_poll_us``, ``drain_sync_us``,
            ``vmm_us``).
        """
        breakdown = _empty_fire_breakdown()
        n_kv_pages = len(plan.page_ids)
        n_mamba_chunks = self._kv_pages_to_mamba_chunks(n_kv_pages)
        use_transfer = self._kv_supports_handle_transfer()
        mamba_static = self._mamba_is_static()

        # S2.5-followup: maybe_execute() normally already ran the headroom
        # check + prepare_*_fire tail-cap synchronously before dispatching
        # here. Only redo it if that didn't happen (e.g. tests/tools that
        # call _execute_locked/_do_vmm directly), so this stays a drop-in
        # replacement for the pre-S2.5-followup behavior.
        if plan.already_prepared:
            breakdown[FIRE_BREAKDOWN_PREPARE] = plan.prepare_us
        else:
            prepare_us = self._check_and_prepare(plan)
            if prepare_us is None:
                return False, breakdown
            breakdown[FIRE_BREAKDOWN_PREPARE] = prepare_us

        if plan.direction == "mamba_to_kv":
            drain_poll_us, drain_sync_us = self._wait_drain("has_capped_mamba_inflight")
            breakdown[FIRE_BREAKDOWN_DRAIN_POLL] = drain_poll_us
            breakdown[FIRE_BREAKDOWN_DRAIN_SYNC] = drain_sync_us

            t_vmm_ns = time.perf_counter_ns()
            if mamba_static:
                # Logical-only Mamba: never touch the Mamba arena.  KV
                # grows from cached handles in its own shared pool.  If
                # there were no prior kv_to_mamba shrinks this grow may
                # exhaust handle availability — that's a config issue
                # (mamba_to_kv first without prior cycle) which we surface
                # via cancellation rather than a corruption.
                try:
                    self.kv_arena.grow(n_kv_pages)
                except Exception as exc:  # noqa: BLE001
                    breakdown[FIRE_BREAKDOWN_VMM] = (
                        time.perf_counter_ns() - t_vmm_ns
                    ) / 1000.0
                    logger.warning(
                        "mamba_to_kv fire skipped (op_id=%d): kv grow failed "
                        "(no handles to re-map?): %s",
                        plan.op_id,
                        exc,
                    )
                    return False, breakdown
                logger.info(
                    "mamba_to_kv VMM done (logical mamba): grew %d kv pages",
                    n_kv_pages,
                )
            elif use_transfer:
                # Legacy path: true handle transfer.  Kept for tests and
                # any deployment where the mamba arena layout supports
                # tail unmap (e.g. a hypothetical slot-major variant).
                raw = self.mamba_arena.shrink_with_handles(n_mamba_chunks)
                balanced = self._balance_handles(raw, n_kv_pages)
                self.kv_arena.grow_with_handles(balanced, n_kv_pages)
                logger.info(
                    "mamba_to_kv VMM done (handle transfer): "
                    "shrunk %d mamba chunks, grew %d kv pages "
                    "(%d handles transferred)",
                    n_mamba_chunks,
                    n_kv_pages,
                    len(balanced),
                )
            else:
                self.mamba_arena.shrink(n_mamba_chunks)
                self.kv_arena.grow(n_kv_pages)
                logger.info(
                    "mamba_to_kv VMM done: shrunk %d mamba chunks, grew %d kv pages",
                    n_mamba_chunks,
                    n_kv_pages,
                )
            breakdown[FIRE_BREAKDOWN_VMM] = (time.perf_counter_ns() - t_vmm_ns) / 1000.0
            return True, breakdown

        elif plan.direction == "kv_to_mamba":
            drain_poll_us, drain_sync_us = self._wait_drain()
            breakdown[FIRE_BREAKDOWN_DRAIN_POLL] = drain_poll_us
            breakdown[FIRE_BREAKDOWN_DRAIN_SYNC] = drain_sync_us

            t_vmm_ns = time.perf_counter_ns()
            if mamba_static:
                # Logical-only Mamba: KV shrinks (handles stay in KV's
                # shared pool for later re-map by mamba_to_kv).  No
                # physical action on the Mamba arena — its tensor view
                # still covers all slots up to base + headroom.
                self.kv_arena.shrink(n_kv_pages)
                logger.info(
                    "kv_to_mamba VMM done (logical mamba): shrunk %d kv pages",
                    n_kv_pages,
                )
            elif use_transfer:
                raw = self.kv_arena.shrink_with_handles(n_kv_pages)
                balanced = self._balance_handles(raw, n_mamba_chunks)
                self.mamba_arena.grow_with_handles(balanced)
                logger.info(
                    "kv_to_mamba VMM done (handle transfer): "
                    "shrunk %d kv pages, grew %d mamba chunks "
                    "(%d handles transferred)",
                    n_kv_pages,
                    n_mamba_chunks,
                    len(balanced),
                )
            else:
                self.kv_arena.shrink(n_kv_pages)
                self.mamba_arena.grow(n_mamba_chunks)
                logger.info(
                    "kv_to_mamba VMM done: shrunk %d kv pages, grew %d mamba chunks",
                    n_kv_pages,
                    n_mamba_chunks,
                )
            breakdown[FIRE_BREAKDOWN_VMM] = (time.perf_counter_ns() - t_vmm_ns) / 1000.0
            return True, breakdown

        else:
            raise ValueError(f"unknown fire direction: {plan.direction}")
