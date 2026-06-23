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

"""Smoke tests for the agentreplay reader and metrics modules.

These tests do not boot an engine; they exercise the trace-parsing and
metric-aggregation surface so the larger A/B harness can fail fast in
CI before it claims a GPU.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from tokenspeed.agentreplay.metrics import PerRequestMetric, aggregate
from tokenspeed.agentreplay.reader import ReplayStep, iter_trace, sessions_in_order
from tokenspeed.agentreplay.replayer import (
    ReplayConfig,
    _compute_session_dispatch_times,
)


_TRACE_DIR = os.environ.get(
    "TOKENSPEED_REPLAY_TRACES",
    "/home/songyang/projects/tokenspeed/dataset/claude-code-traces/traces",
)


def _has_real_trace() -> bool:
    return os.path.exists(os.path.join(_TRACE_DIR, "cc_qwen_mamba.jsonl"))


def _write_synthetic_trace(path: str) -> None:
    lines = [
        # Root session with two steps.
        {
            "t": 0.0,
            "program_id": "sess-a",
            "step": 1,
            "parent_program_id": None,
            "spawn_ts": None,
            "input_ids": [1, 2, 3, 4],
            "forced_output_ids": [10, 11],
            "tool_gap_after": 0.5,
        },
        {
            "t": 0.7,
            "program_id": "sess-a",
            "step": 2,
            "parent_program_id": None,
            "spawn_ts": None,
            "input_ids": [1, 2, 3, 4, 10, 11, 5],
            "forced_output_ids": [12],
            "tool_gap_after": 0.0,
        },
        # Subagent of sess-a.
        {
            "t": 0.3,
            "program_id": "sess-a-sub",
            "step": 1,
            "parent_program_id": "sess-a",
            "spawn_ts": 0.25,
            "input_ids": [7, 8],
            "forced_output_ids": [20, 21, 22],
            "tool_gap_after": 0.0,
        },
    ]
    with open(path, "w") as fh:
        for obj in lines:
            fh.write(json.dumps(obj) + "\n")


def test_reader_groups_sessions_in_order():
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tmp:
        path = tmp.name
    try:
        _write_synthetic_trace(path)
        groups = sessions_in_order(path)
        assert [pid for pid, _ in groups] == ["sess-a", "sess-a-sub"]
        a_steps = dict(groups)["sess-a"]
        assert [s.step for s in a_steps] == [1, 2]
        sub_steps = dict(groups)["sess-a-sub"]
        assert sub_steps[0].is_subagent is True
        assert sub_steps[0].parent_program_id == "sess-a"
        assert sub_steps[0].spawn_ts == 0.25
        # ``input_ids`` and ``forced_output_ids`` must survive verbatim;
        # the harness relies on these being trace-exact to give a fair
        # A/B comparison.
        assert a_steps[0].input_ids == [1, 2, 3, 4]
        assert a_steps[0].forced_output_ids == [10, 11]
    finally:
        os.unlink(path)


def test_aggregate_handles_failures_separately():
    records = [
        PerRequestMetric(
            rid="ok-1",
            program_id="p1",
            step=1,
            parent_program_id=None,
            arrival_t=0.0,
            first_token_t=0.05,
            finish_t=0.5,
            prompt_tokens=100,
            output_tokens=10,
            cached_tokens=20,
            finish_reason="length",
        ),
        PerRequestMetric(
            rid="fail-1",
            program_id="p1",
            step=2,
            parent_program_id=None,
            arrival_t=0.6,
            first_token_t=-1,
            finish_t=-1,
            prompt_tokens=100,
            output_tokens=0,
            cached_tokens=0,
            finish_reason="",
            error="timeout_after_30s",
        ),
    ]
    summary = aggregate(records, wall_start=0.0, wall_end=1.0)
    assert summary["n_requests"] == 2
    assert summary["n_finished"] == 1
    assert summary["n_failed"] == 1
    # TTFT/throughput should only reflect the successful request.
    assert summary["ttft_ms_p50"] == pytest.approx(50.0)
    assert summary["output_tokens_per_s"] == pytest.approx(10.0)
    assert summary["cache_hit_ratio"] == pytest.approx(0.2)


def _mk_step(pid: str, t: float, spawn: float | None = None) -> ReplayStep:
    return ReplayStep(
        t=t,
        program_id=pid,
        step=1,
        parent_program_id=None,
        spawn_ts=spawn,
        input_ids=[0],
        forced_output_ids=[1],
        tool_gap_after=0.0,
    )


def test_compute_dispatch_times_normalizes_and_caps_gaps():
    sessions = [
        ("s0", [_mk_step("s0", t=1_000.0)]),
        ("s1", [_mk_step("s1", t=1_000_002.0)]),
        ("s2", [_mk_step("s2", t=1_000_004.0)]),
    ]
    cfg = ReplayConfig(
        trace_path="x", normalize_time=True, max_inter_session_gap_s=5.0
    )
    out = _compute_session_dispatch_times(sessions, cfg)
    # First session arrives at t=0 after normalization. Second is
    # ~1e6 seconds later in raw time -> capped to 5s. Third is 2s after
    # second in raw time (under the cap), so it stays.
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(5.0)
    assert out[2] == pytest.approx(7.0)


def test_compute_dispatch_times_respects_subagent_spawn_ts():
    sessions = [
        ("parent", [_mk_step("parent", t=10.0)]),
        ("sub", [_mk_step("sub", t=99.0, spawn=12.0)]),
    ]
    cfg = ReplayConfig(
        trace_path="x", normalize_time=True, max_inter_session_gap_s=None
    )
    out = _compute_session_dispatch_times(sessions, cfg)
    # Sub arrives 2s after parent (spawn_ts honored, not step.t).
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(2.0)


def test_compute_dispatch_times_applies_time_scale():
    sessions = [
        ("s0", [_mk_step("s0", t=0.0)]),
        ("s1", [_mk_step("s1", t=10.0)]),
    ]
    cfg = ReplayConfig(trace_path="x", time_scale=2.0, max_inter_session_gap_s=None)
    out = _compute_session_dispatch_times(sessions, cfg)
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(5.0)


@pytest.mark.skipif(not _has_real_trace(), reason="dataset trace not present")
def test_reader_parses_real_cc_qwen_mamba_trace():
    """Sanity check against the real ``cc_qwen_mamba.jsonl`` (900 steps)."""
    path = os.path.join(_TRACE_DIR, "cc_qwen_mamba.jsonl")
    steps = list(iter_trace(path))
    assert len(steps) == 900
    sessions = sessions_in_order(path)
    # Expect ~300 sessions and ~200 subagents per dataset card.
    subagent_count = sum(1 for _, ss in sessions if any(s.is_subagent for s in ss))
    assert 280 <= len(sessions) <= 320
    assert 150 <= subagent_count <= 250
