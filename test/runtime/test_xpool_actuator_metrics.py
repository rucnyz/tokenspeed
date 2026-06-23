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

"""Tests for the per-direction fire counters on :class:`XPoolActuator`.

The counters back the replay-time-series log (``budget.jsonl``) so the
HiMA A/B harness can distinguish a *real* committed cross-pool transfer
from a *cancelled* one. We exercise the counter increments directly via
the actuator's ``_execute_locked`` path, stubbing out the heavy VMM and
scheduler bookkeeping.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from tokenspeed.runtime.cache.arena.xpool_actuator import FirePlan, XPoolActuator


def _make_actuator(*, vmm_succeeds: bool) -> tuple[XPoolActuator, MagicMock]:
    scheduler = MagicMock()
    scheduler.cancel_xpool_fire = MagicMock()
    actuator = XPoolActuator(
        kv_arena=MagicMock(),
        mamba_arena=MagicMock(),
        scheduler=scheduler,
        kv_bytes_per_page=4096,
    )
    actuator._do_vmm = MagicMock(return_value=vmm_succeeds)
    return actuator, scheduler


def _plan(direction: str, op_id: int) -> FirePlan:
    return FirePlan(
        direction=direction,
        page_ids=[1, 2, 3, 4],
        op_id=op_id,
        cpp_plan=object(),
    )


def test_committed_fire_increments_correct_direction():
    actuator, _ = _make_actuator(vmm_succeeds=True)
    actuator._execute_locked(_plan("kv_to_mamba", op_id=1))
    actuator._execute_locked(_plan("mamba_to_kv", op_id=2))
    actuator._execute_locked(_plan("kv_to_mamba", op_id=3))

    assert actuator.committed_kv_to_mamba == 2
    assert actuator.committed_mamba_to_kv == 1
    assert actuator.cancelled_kv_to_mamba == 0
    assert actuator.cancelled_mamba_to_kv == 0


def test_cancelled_fire_increments_only_cancelled_counters():
    actuator, scheduler = _make_actuator(vmm_succeeds=False)
    actuator._execute_locked(_plan("kv_to_mamba", op_id=1))
    actuator._execute_locked(_plan("mamba_to_kv", op_id=2))

    assert actuator.committed_kv_to_mamba == 0
    assert actuator.committed_mamba_to_kv == 0
    assert actuator.cancelled_kv_to_mamba == 1
    assert actuator.cancelled_mamba_to_kv == 1
    # Scheduler.cancel_xpool_fire must be invoked exactly once per cancelled
    # plan so the budgeter latch is released.
    assert scheduler.cancel_xpool_fire.call_count == 2
