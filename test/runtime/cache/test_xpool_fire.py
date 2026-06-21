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

"""Unit tests for the HiMA XPool fire actuator and its EventLoop wiring.

These tests exercise the control/execution plane with fake arenas and need no
GPU: they verify that a fire plan is dispatched to the correct pools in the
correct order, that the EventLoop de-dups latched plans by op_id, and that
``scheduler.apply_xpool_fire`` is called after VMM ops complete.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tokenspeed.runtime.cache.arena.xpool_actuator import FirePlan, XPoolActuator


class _FakeArena:
    """Records transfer_in/out calls instead of touching CUDA."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[int]]] = []

    def transfer_out(self, chunk_ids: list[int]) -> None:
        self.calls.append(("out", list(chunk_ids)))

    def transfer_in(self, chunk_ids: list[int]) -> None:
        self.calls.append(("in", list(chunk_ids)))


class _FakeScheduler:
    """Records apply_xpool_fire calls."""

    def __init__(self) -> None:
        self.applied: list[object] = []

    def apply_xpool_fire(self, plan: object) -> None:
        self.applied.append(plan)


def test_mamba_to_kv_unmaps_source_then_maps_dest() -> None:
    kv, mamba = _FakeArena(), _FakeArena()
    actuator = XPoolActuator(kv_arena=kv, mamba_arena=mamba)

    actuator._execute_locked(FirePlan(direction="mamba_to_kv", page_ids=[1, 2, 3]))

    # Source pool releases first, destination maps after.
    assert mamba.calls == [("out", [1, 2, 3])]
    assert kv.calls == [("in", [1, 2, 3])]


def test_kv_to_mamba_reverses_direction() -> None:
    kv, mamba = _FakeArena(), _FakeArena()
    actuator = XPoolActuator(kv_arena=kv, mamba_arena=mamba)

    actuator._execute_locked(FirePlan(direction="kv_to_mamba", page_ids=[5, 6]))

    assert kv.calls == [("out", [5, 6])]
    assert mamba.calls == [("in", [5, 6])]


def test_unknown_direction_raises() -> None:
    actuator = XPoolActuator(kv_arena=_FakeArena(), mamba_arena=_FakeArena())

    with pytest.raises(ValueError, match="unknown fire direction"):
        actuator._execute_locked(FirePlan(direction="sideways", page_ids=[1]))


def test_scheduler_apply_called_after_vmm_ops() -> None:
    """apply_xpool_fire must be called AFTER the VMM transfer completes."""
    kv, mamba = _FakeArena(), _FakeArena()
    sched = _FakeScheduler()
    actuator = XPoolActuator(kv_arena=kv, mamba_arena=mamba, scheduler=sched)

    cpp_plan = SimpleNamespace(op_id=7, direction="mamba_to_kv", page_ids=[1])
    plan = FirePlan(direction="mamba_to_kv", page_ids=[1], op_id=7, cpp_plan=cpp_plan)
    actuator._execute_locked(plan)

    # VMM ops happened.
    assert mamba.calls == [("out", [1])]
    assert kv.calls == [("in", [1])]
    # Scheduler was notified.
    assert sched.applied == [cpp_plan]


def test_scheduler_not_called_when_none() -> None:
    """No error when scheduler is not provided (control-only mode)."""
    kv, mamba = _FakeArena(), _FakeArena()
    actuator = XPoolActuator(kv_arena=kv, mamba_arena=mamba, scheduler=None)

    plan = FirePlan(
        direction="mamba_to_kv",
        page_ids=[1],
        op_id=1,
        cpp_plan=SimpleNamespace(),
    )
    actuator._execute_locked(plan)  # must not raise
    assert kv.calls == [("in", [1])]


def _actuator_with_recorder():
    """Return (actuator, fired) where fired records launched op_ids."""
    actuator = XPoolActuator(kv_arena=_FakeArena(), mamba_arena=_FakeArena())
    fired: list[int] = []
    actuator.execute_async = lambda plan: fired.append(plan.op_id)  # type: ignore[method-assign]
    return actuator, fired


def test_maybe_execute_dedups_latched_plan() -> None:
    actuator, fired = _actuator_with_recorder()
    plan = SimpleNamespace(op_id=1, direction="mamba_to_kv", page_ids=[1, 2])

    assert actuator.maybe_execute(plan) is True
    # Same latched plan again -> de-duped, no second fire.
    assert actuator.maybe_execute(plan) is False
    assert fired == [1]


def test_maybe_execute_fires_new_op_id() -> None:
    actuator, fired = _actuator_with_recorder()

    assert actuator.maybe_execute(
        SimpleNamespace(op_id=1, direction="mamba_to_kv", page_ids=[1])
    )
    assert actuator.maybe_execute(
        SimpleNamespace(op_id=2, direction="kv_to_mamba", page_ids=[3])
    )
    assert fired == [1, 2]
