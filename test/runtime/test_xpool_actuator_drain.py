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
        actuator._wait_drain()

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
        actuator._wait_drain()

    assert mock_sync.call_count == 2
    assert mock_sync.call_args_list[0].args == (actuator._device,)
    mock_sleep.assert_called_once()  # only the first True triggers a poll sleep


def test_wait_drain_noop_without_scheduler():
    actuator = XPoolActuator(kv_arena=MagicMock(), mamba_arena=MagicMock(), scheduler=None)
    with patch("torch.cuda.synchronize") as mock_sync:
        actuator._wait_drain()
    mock_sync.assert_not_called()

def test_wait_drain_noop_without_drain_fn():
    scheduler = MagicMock(spec=[])  # no has_capped_kv_inflight attribute
    actuator = XPoolActuator(
        kv_arena=MagicMock(), mamba_arena=MagicMock(), scheduler=scheduler
    )
    with patch("torch.cuda.synchronize") as mock_sync:
        actuator._wait_drain()
    mock_sync.assert_not_called()
