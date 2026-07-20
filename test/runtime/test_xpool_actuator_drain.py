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

"""Tests for :meth:`XPoolActuator._wait_drain`.

These guard the poll -> sync -> recheck loop added to close a race where a
request's KV pages are marked "drained" by the C++ scheduler microseconds
before we ``torch.cuda.synchronize()``, but its terminal decode/prefill
kernel -- submitted ahead of time by the overlap event loop
(``event_loop_overlap``) -- is still sitting on the execution stream. A
naive poll-then-sync-once implementation can unmap those pages while that
kernel is still in flight, producing a CUDA "illegal memory access". See
the ``sys_lru`` HiMA ablation crash this was written to fix.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from tokenspeed.runtime.cache.arena.xpool_actuator import XPoolActuator


def _make_actuator(*, device: int | None = 3) -> tuple[XPoolActuator, MagicMock]:
    scheduler = MagicMock()
    mamba_arena = MagicMock()
    mamba_arena._device = device
    actuator = XPoolActuator(
        kv_arena=MagicMock(),
        mamba_arena=mamba_arena,
        scheduler=scheduler,
    )
    return actuator, scheduler


def test_wait_drain_picks_up_device_from_mamba_arena():
    actuator, _ = _make_actuator(device=3)
    assert actuator._device == 3


def test_wait_drain_syncs_once_when_drain_clears_immediately():
    """The common case: no in-flight capped pages -> exactly one sync."""
    actuator, scheduler = _make_actuator()
    scheduler.has_capped_kv_inflight = MagicMock(return_value=False)

    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.synchronize"
    ) as mock_sync:
        drained, poll_us, sync_us = actuator._wait_drain()

    assert drained is True
    assert poll_us >= 0.0
    assert sync_us >= 0.0
    mock_sync.assert_called_once_with(actuator._device)
    # One poll (False -> exit poll loop) + one post-sync recheck (also
    # False -> done): two calls total for the clean, no-contention case.
    assert scheduler.has_capped_kv_inflight.call_count == 2


def test_wait_drain_rechecks_after_sync_when_new_inflight_appears():
    """If the drain check flips back to True right after a sync (a request's
    terminal kernel was still in flight when we synced), we must loop back
    and sync again rather than proceeding straight to unmap."""
    actuator, scheduler = _make_actuator()
    # Sequence: True (still draining) -> False (poll loop exits, sync #1) ->
    # recheck True (something raced back in) -> False (poll loop exits again,
    # sync #2) -> recheck False -> done.
    scheduler.has_capped_kv_inflight = MagicMock(
        side_effect=[True, False, True, False, False]
    )

    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.synchronize"
    ) as mock_sync, patch("time.sleep") as mock_sleep:
        drained, poll_us, sync_us = actuator._wait_drain()

    assert drained is True
    assert poll_us >= 0.0
    assert sync_us >= 0.0
    assert mock_sync.call_count == 2
    assert mock_sync.call_args_list[0].args == (actuator._device,)
    mock_sleep.assert_called_once()  # only the first True triggers a poll sleep


def test_wait_drain_noop_without_scheduler():
    actuator = XPoolActuator(kv_arena=MagicMock(), mamba_arena=MagicMock(), scheduler=None)
    with patch("torch.cuda.synchronize") as mock_sync:
        drained, poll_us, sync_us = actuator._wait_drain()
    assert drained is True
    assert poll_us == 0.0
    assert sync_us == 0.0
    mock_sync.assert_not_called()


def test_wait_drain_noop_without_drain_fn():
    scheduler = MagicMock(spec=[])  # no has_capped_kv_inflight attribute
    actuator = XPoolActuator(
        kv_arena=MagicMock(), mamba_arena=MagicMock(), scheduler=scheduler
    )
    with patch("torch.cuda.synchronize") as mock_sync:
        drained, poll_us, sync_us = actuator._wait_drain()
    assert drained is True
    assert poll_us == 0.0
    assert sync_us == 0.0
    mock_sync.assert_not_called()


def test_wait_drain_timeout_reports_not_drained_and_never_unmaps():
    """Regression for the sustained-KV-scarcity crash (cc_qwen_t6 @
    max_total_tokens=640000): when long-running decode requests hold capped
    pages past DRAIN_TIMEOUT_S, _wait_drain must report drained=False so the
    fire is cancelled. The old behavior returned normally ("proceeding with
    unmap"), and the subsequent cuMemUnmap of still-in-use pages surfaced a
    CUDA illegal memory access one decode step later, killing the engine."""
    actuator, scheduler = _make_actuator()
    scheduler.has_capped_kv_inflight = MagicMock(return_value=True)
    actuator.DRAIN_TIMEOUT_S = 0.01

    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.synchronize"
    ):
        drained, poll_us, sync_us = actuator._wait_drain()

    assert drained is False


def test_wait_drain_aborts_early_when_count_stalls():
    """No-progress early abort (2026-07-19): when the scheduler exposes a
    capped-inflight *count* that never decreases (the tail is pinned by
    long-running decode requests), _wait_drain must cancel via _NoProgress
    well before DRAIN_TIMEOUT_S, so the prepare tail-cap is released promptly
    instead of depressing capacity for the full timeout."""
    actuator, scheduler = _make_actuator()
    scheduler.has_capped_kv_inflight = MagicMock(return_value=True)
    scheduler.capped_kv_inflight_count = MagicMock(return_value=4)  # never drops
    actuator.DRAIN_TIMEOUT_S = 100.0  # would otherwise hang ~forever
    actuator.DRAIN_NOPROGRESS_ABORT_S = 0.02

    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.synchronize"
    ):
        t0 = time.monotonic()
        drained, _poll_us, _sync_us = actuator._wait_drain()
        elapsed = time.monotonic() - t0

    assert drained is False
    # Aborted via the sub-second no-progress window, NOT the 100 s timeout.
    assert elapsed < 5.0


def test_wait_drain_completes_when_count_decreases_to_zero():
    """A genuinely-draining fire (count strictly decreasing) is unaffected by
    the no-progress abort and completes normally."""
    actuator, scheduler = _make_actuator()
    scheduler.capped_kv_inflight_count = MagicMock(side_effect=[3, 2, 1, 0, 0])
    actuator.DRAIN_NOPROGRESS_ABORT_S = 10.0  # generous; must not trip

    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.synchronize"
    ) as mock_sync, patch("time.sleep"):
        drained, _poll_us, _sync_us = actuator._wait_drain()

    assert drained is True
    mock_sync.assert_called_once_with(actuator._device)


def test_wait_drain_noprogress_disabled_when_window_zero():
    """DRAIN_NOPROGRESS_ABORT_S=0 disables the early abort: a stalled count
    then falls through to the DRAIN_TIMEOUT_S backstop instead."""
    actuator, scheduler = _make_actuator()
    scheduler.capped_kv_inflight_count = MagicMock(return_value=2)  # stalled
    actuator.DRAIN_NOPROGRESS_ABORT_S = 0.0
    actuator.DRAIN_TIMEOUT_S = 0.02

    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.synchronize"
    ):
        drained, _poll_us, _sync_us = actuator._wait_drain()

    assert drained is False  # via timeout, not no-progress


def test_constructor_overrides_drain_tunables():
    """The regime-matrix driver sweeps these per arm via env vars, which
    EngineCore forwards as constructor kwargs; unset ones keep class defaults."""
    tuned = XPoolActuator(
        kv_arena=MagicMock(),
        mamba_arena=MagicMock(),
        scheduler=None,
        drain_timeout_s=1.5,
        drain_poll_s=0.001,
        timeout_cooldown_s=7.0,
        drain_noprogress_abort_s=0.3,
    )
    assert tuned.DRAIN_TIMEOUT_S == 1.5
    assert tuned.DRAIN_POLL_S == 0.001
    assert tuned.TIMEOUT_COOLDOWN_S == 7.0
    assert tuned.DRAIN_NOPROGRESS_ABORT_S == 0.3

    default = XPoolActuator(kv_arena=MagicMock(), mamba_arena=MagicMock())
    assert default.DRAIN_TIMEOUT_S == XPoolActuator.DRAIN_TIMEOUT_S
    assert default.DRAIN_NOPROGRESS_ABORT_S == XPoolActuator.DRAIN_NOPROGRESS_ABORT_S


def _plan_stub(op_id: int, direction: str = "kv_to_mamba"):
    plan = MagicMock()
    plan.op_id = op_id
    plan.direction = direction
    plan.page_ids = [1, 2, 3]
    return plan


def test_maybe_execute_skips_plan_while_fire_outstanding():
    """Regression for the 2026-07-18 long_horizon run (1375 dispatched / 0
    committed fires): the budgeter emits a plan every tick while one drain
    wait can take DRAIN_TIMEOUT_S, so without backpressure each tick capped
    another tail slice via prepare_*_fire and queued another worker thread.
    While a fire is outstanding, new plans must be skipped WITHOUT touching
    scheduler state (no prepare, no cancel -- cancel would undo the
    outstanding fire's prepare caps mid-drain)."""
    actuator, scheduler = _make_actuator()
    actuator._outstanding_fires = 1

    with patch.object(actuator, "_check_and_prepare") as mock_prepare, patch.object(
        actuator, "execute_async"
    ) as mock_spawn:
        handled = actuator.maybe_execute(_plan_stub(op_id=7))

    assert handled is True
    assert actuator.skipped_busy == 1
    mock_prepare.assert_not_called()
    mock_spawn.assert_not_called()
    scheduler.cancel_xpool_fire.assert_not_called()
    scheduler.prepare_kv_to_mamba_fire.assert_not_called()


def test_maybe_execute_skips_direction_during_cooldown():
    """After a cancelled fire arms the per-direction cooldown, new plans in
    that direction are skipped (no prepare / no thread) until it expires;
    the opposite direction stays unaffected."""
    actuator, scheduler = _make_actuator()
    actuator._arm_direction_cooldown("mamba_to_kv")

    with patch.object(actuator, "_check_and_prepare") as mock_prepare, patch.object(
        actuator, "execute_async"
    ):
        handled = actuator.maybe_execute(_plan_stub(op_id=8, direction="mamba_to_kv"))
        assert handled is True
        assert actuator.skipped_cooldown == 1
        mock_prepare.assert_not_called()

        # Opposite direction is not gated by mamba_to_kv's cooldown.
        mock_prepare.return_value = 0.0
        actuator.maybe_execute(_plan_stub(op_id=9, direction="kv_to_mamba"))
        mock_prepare.assert_called_once()


def test_cancelled_fire_arms_cooldown_and_decrements_outstanding():
    """A drain-timeout cancel inside the worker must arm the cooldown for
    that direction and release the outstanding-fire gate."""
    from tokenspeed.runtime.cache.arena.xpool_actuator import FirePlan

    actuator, scheduler = _make_actuator()
    # Non-static mamba arena (max != mapped) so the physical-unmap drain path
    # actually runs -- the drain is skipped entirely for a static (logical
    # only) arena, where there is no unmap race to wait on.
    actuator.mamba_arena.max_chunks = 200
    actuator.mamba_arena.mapped_chunks = 100
    scheduler.has_capped_mamba_inflight = MagicMock(return_value=True)
    actuator.DRAIN_TIMEOUT_S = 0.01
    actuator._outstanding_fires = 1

    plan = FirePlan(
        op_id=2,
        direction="mamba_to_kv",
        page_ids=[1, 2],
        cpp_plan=MagicMock(),
        already_prepared=True,
        prepare_us=0.0,
    )
    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.synchronize"
    ):
        actuator._execute_locked(plan)

    assert actuator.cancelled_mamba_to_kv == 1
    assert actuator._outstanding_fires == 0
    assert "mamba_to_kv" in actuator._direction_cooldown_until
    scheduler.cancel_xpool_fire.assert_called_once()


def test_do_vmm_cancels_fire_when_drain_times_out():
    """End-to-end through _do_vmm: a timed-out drain must return
    committed=False (-> _execute_locked cancels via cancel_xpool_fire, which
    undoes the prepare Shrink) and must never touch the arenas."""
    from tokenspeed.runtime.cache.arena.xpool_actuator import FirePlan

    actuator, scheduler = _make_actuator()
    scheduler.has_capped_kv_inflight = MagicMock(return_value=True)
    actuator.DRAIN_TIMEOUT_S = 0.01

    plan = FirePlan(
        op_id=1,
        direction="kv_to_mamba",
        page_ids=[1, 2, 3],
        cpp_plan=MagicMock(),
        already_prepared=True,
        prepare_us=0.0,
    )

    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.synchronize"
    ):
        committed, breakdown = actuator._do_vmm(plan)

    assert committed is False
    actuator.kv_arena.shrink.assert_not_called()
    actuator.kv_arena.shrink_with_handles.assert_not_called()
    actuator.mamba_arena.grow.assert_not_called()
