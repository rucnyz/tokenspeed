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


class _StaticMambaArena:
    """Mamba arena pre-mapped to its full extent (mapped == max).

    Mirrors the production layout enforced by HiMA Phase 3
    (S2.2-followup): the conv/ssm contiguous-slice tensor lives on a
    pre-mapped VMM window that the actuator must never resize.  Calls
    to ``grow``/``shrink`` here would indicate a regression to the
    crash-prone post-boot resize path.
    """

    def __init__(self, max_chunks: int = 100) -> None:
        self.max_chunks = max_chunks
        self.mapped_chunks = max_chunks  # static
        self.calls: list[tuple[str, int]] = []

    def grow(self, n: int) -> None:  # pragma: no cover - guard against regression
        self.calls.append(("grow", n))
        raise AssertionError("static mamba arena must not be resized post-boot")

    def shrink(self, n: int) -> None:  # pragma: no cover - guard against regression
        self.calls.append(("shrink", n))
        raise AssertionError("static mamba arena must not be resized post-boot")

    def grow_with_handles(self, handles):  # pragma: no cover
        self.calls.append(("grow_with_handles", len(handles)))
        raise AssertionError("static mamba arena must not be resized post-boot")

    def shrink_with_handles(self, n: int):  # pragma: no cover
        self.calls.append(("shrink_with_handles", n))
        raise AssertionError("static mamba arena must not be resized post-boot")


class _KvArena:
    """Records grow/shrink calls; reports configurable headroom."""

    def __init__(self, mapped_chunks: int = 100, max_chunks: int = 200) -> None:
        self.max_chunks = max_chunks
        self.mapped_chunks = mapped_chunks
        self.calls: list[tuple[str, int]] = []

    @property
    def headroom_pages(self) -> int:
        # 1 page == 1 chunk in this stub (the actuator only uses the value
        # for the headroom check; it doesn't care about unit math here).
        return self.max_chunks - self.mapped_chunks

    def grow(self, n: int) -> None:
        self.mapped_chunks = min(self.max_chunks, self.mapped_chunks + n)
        self.calls.append(("grow", n))

    def shrink(self, n: int) -> None:
        self.mapped_chunks = max(0, self.mapped_chunks - n)
        self.calls.append(("shrink", n))


class _FakeScheduler:
    """Records lifecycle calls: prepare/drain/apply/cancel."""

    def __init__(self) -> None:
        self.applied: list[object] = []
        self.cancelled: int = 0
        self.prepared: list[tuple[str, int]] = []

    def apply_xpool_fire(self, plan: object) -> None:
        self.applied.append(plan)

    def cancel_xpool_fire(self) -> None:
        self.cancelled += 1

    def prepare_kv_to_mamba_fire(self, n: int) -> None:
        self.prepared.append(("kv_to_mamba", n))

    def prepare_mamba_to_kv_fire(self, n: int) -> None:
        self.prepared.append(("mamba_to_kv", n))

    def has_capped_kv_inflight(self) -> bool:
        return False

    def has_capped_mamba_inflight(self) -> bool:
        return False


def test_static_mamba_kv_to_mamba_shrinks_kv_only() -> None:
    """kv_to_mamba: KV physically shrinks, Mamba stays static."""
    kv = _KvArena(mapped_chunks=10, max_chunks=20)
    mamba = _StaticMambaArena()
    sched = _FakeScheduler()
    actuator = XPoolActuator(kv_arena=kv, mamba_arena=mamba, scheduler=sched)

    cpp_plan = SimpleNamespace(op_id=1, direction="kv_to_mamba", page_ids=[1, 2, 3])
    plan = FirePlan(
        direction="kv_to_mamba", page_ids=[1, 2, 3], op_id=1, cpp_plan=cpp_plan
    )
    actuator._execute_locked(plan)

    assert kv.calls == [("shrink", 3)]
    assert mamba.calls == []  # mamba arena untouched
    assert sched.applied == [cpp_plan]  # logical fire committed


def test_static_mamba_to_kv_grows_kv_only() -> None:
    """mamba_to_kv: KV physically grows from cached handles, Mamba stays static."""
    kv = _KvArena(mapped_chunks=5, max_chunks=20)  # has slack to grow
    mamba = _StaticMambaArena()
    sched = _FakeScheduler()
    actuator = XPoolActuator(kv_arena=kv, mamba_arena=mamba, scheduler=sched)

    cpp_plan = SimpleNamespace(op_id=1, direction="mamba_to_kv", page_ids=[1, 2])
    plan = FirePlan(
        direction="mamba_to_kv", page_ids=[1, 2], op_id=1, cpp_plan=cpp_plan
    )
    actuator._execute_locked(plan)

    assert kv.calls == [("grow", 2)]
    assert mamba.calls == []
    assert sched.applied == [cpp_plan]


def test_mamba_to_kv_cancels_when_kv_full() -> None:
    """mamba_to_kv must cancel when KV arena cannot grow further."""
    kv = _KvArena(mapped_chunks=20, max_chunks=20)  # no slack
    mamba = _StaticMambaArena()
    sched = _FakeScheduler()
    actuator = XPoolActuator(kv_arena=kv, mamba_arena=mamba, scheduler=sched)

    cpp_plan = SimpleNamespace(op_id=2, direction="mamba_to_kv", page_ids=[1])
    plan = FirePlan(direction="mamba_to_kv", page_ids=[1], op_id=2, cpp_plan=cpp_plan)
    actuator._execute_locked(plan)

    assert kv.calls == []  # no physical action on cancel
    assert sched.applied == []
    assert sched.cancelled == 1


def test_unknown_direction_raises() -> None:
    # _execute_locked logs and swallows exceptions for background-thread
    # safety, so we test _do_vmm directly.
    actuator = XPoolActuator(kv_arena=_KvArena(), mamba_arena=_StaticMambaArena())

    with pytest.raises(ValueError, match="unknown fire direction"):
        actuator._do_vmm(FirePlan(direction="sideways", page_ids=[1]))


def test_scheduler_apply_called_after_vmm_ops() -> None:
    """apply_xpool_fire must be called AFTER the VMM transfer completes."""
    kv = _KvArena(mapped_chunks=5, max_chunks=20)
    mamba = _StaticMambaArena()
    sched = _FakeScheduler()
    actuator = XPoolActuator(kv_arena=kv, mamba_arena=mamba, scheduler=sched)

    cpp_plan = SimpleNamespace(op_id=7, direction="mamba_to_kv", page_ids=[1])
    plan = FirePlan(direction="mamba_to_kv", page_ids=[1], op_id=7, cpp_plan=cpp_plan)
    actuator._execute_locked(plan)

    assert kv.calls == [("grow", 1)]
    assert sched.applied == [cpp_plan]


def test_record_fire_cost_seeds_then_smooths_ewma() -> None:
    """First sample seeds the EWMA so we don't sit at 0 forever."""
    actuator = XPoolActuator(kv_arena=_KvArena(), mamba_arena=_StaticMambaArena())

    actuator._record_fire_cost(n_pages=10, elapsed_us=1000.0)
    assert actuator.ewma_xfer_us_per_page == pytest.approx(100.0)
    assert actuator.last_fire_us == pytest.approx(1000.0)
    assert actuator.last_fire_pages == 10

    actuator._record_fire_cost(n_pages=10, elapsed_us=2000.0)
    expected = (1.0 - 0.25) * 100.0 + 0.25 * 200.0
    assert actuator.ewma_xfer_us_per_page == pytest.approx(expected)


def test_record_fire_cost_ignores_invalid_samples() -> None:
    actuator = XPoolActuator(kv_arena=_KvArena(), mamba_arena=_StaticMambaArena())
    actuator._record_fire_cost(n_pages=0, elapsed_us=10.0)
    actuator._record_fire_cost(n_pages=10, elapsed_us=0.0)
    assert actuator.ewma_xfer_us_per_page == 0.0


def test_committed_fire_updates_ewma() -> None:
    """Full path: _execute_locked must time and record the VMM cost."""
    kv = _KvArena(mapped_chunks=5, max_chunks=20)
    mamba = _StaticMambaArena()
    sched = _FakeScheduler()
    actuator = XPoolActuator(kv_arena=kv, mamba_arena=mamba, scheduler=sched)

    cpp_plan = SimpleNamespace(op_id=11, direction="mamba_to_kv", page_ids=[1, 2])
    plan = FirePlan(
        direction="mamba_to_kv", page_ids=[1, 2], op_id=11, cpp_plan=cpp_plan
    )
    actuator._execute_locked(plan)
    assert sched.applied == [cpp_plan]
    assert actuator.ewma_xfer_us_per_page > 0.0
    assert actuator.last_fire_pages == 2


def test_cancelled_fire_does_not_update_ewma() -> None:
    """Cancelled fires (no VMM work) must leave the EWMA at zero."""
    kv = _KvArena(mapped_chunks=20, max_chunks=20)  # no slack -> cancel
    mamba = _StaticMambaArena()
    sched = _FakeScheduler()
    actuator = XPoolActuator(kv_arena=kv, mamba_arena=mamba, scheduler=sched)

    cpp_plan = SimpleNamespace(op_id=12, direction="mamba_to_kv", page_ids=[1])
    plan = FirePlan(direction="mamba_to_kv", page_ids=[1], op_id=12, cpp_plan=cpp_plan)
    actuator._execute_locked(plan)
    assert sched.cancelled == 1
    assert actuator.ewma_xfer_us_per_page == 0.0


def test_scheduler_not_called_when_none() -> None:
    """No error when scheduler is not provided (control-only mode)."""
    kv = _KvArena(mapped_chunks=5, max_chunks=20)
    mamba = _StaticMambaArena()
    actuator = XPoolActuator(kv_arena=kv, mamba_arena=mamba, scheduler=None)

    plan = FirePlan(
        direction="mamba_to_kv",
        page_ids=[1],
        op_id=1,
        cpp_plan=SimpleNamespace(),
    )
    actuator._execute_locked(plan)  # must not raise
    assert kv.calls == [("grow", 1)]


def _actuator_with_recorder():
    """Return (actuator, fired) where fired records launched op_ids."""
    actuator = XPoolActuator(kv_arena=_KvArena(), mamba_arena=_StaticMambaArena())
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


def test_static_mamba_signal_detection() -> None:
    """_mamba_is_static True iff arena.max_chunks == arena.mapped_chunks."""
    actuator = XPoolActuator(
        kv_arena=_KvArena(), mamba_arena=_StaticMambaArena(max_chunks=42)
    )
    assert actuator._mamba_is_static() is True

    # Construct a dynamic stand-in (mapped < max).
    dynamic = SimpleNamespace(max_chunks=10, mapped_chunks=5)
    actuator.mamba_arena = dynamic  # type: ignore[assignment]
    assert actuator._mamba_is_static() is False
