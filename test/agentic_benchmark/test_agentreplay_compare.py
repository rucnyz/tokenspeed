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

"""Tests for ``tokenspeed.agentreplay.compare`` aggregation logic."""

from __future__ import annotations

import json
import os

import pytest

from tokenspeed.agentreplay.compare import _find_summaries, _read_arm


def _write_summary(path: str, **fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(fields, fh)


def test_find_summaries_accepts_leaf_dir(tmp_path):
    leaf = tmp_path / "run0"
    _write_summary(str(leaf / "summary.json"), ttft_ms_p50=12.0)
    found = _find_summaries(str(leaf))
    assert len(found) == 1
    assert found[0].endswith("summary.json")


def test_find_summaries_walks_rep_subdirs(tmp_path):
    parent = tmp_path / "base"
    _write_summary(str(parent / "rep0" / "summary.json"), ttft_ms_p50=10.0)
    _write_summary(str(parent / "rep1" / "summary.json"), ttft_ms_p50=12.0)
    _write_summary(str(parent / "rep2" / "summary.json"), ttft_ms_p50=8.0)
    found = _find_summaries(str(parent))
    assert len(found) == 3


def test_read_arm_averages_across_reps(tmp_path):
    parent = tmp_path / "sys"
    _write_summary(
        str(parent / "rep0" / "summary.json"),
        ttft_ms_p50=10.0,
        output_tokens_per_s=100.0,
        cache_hit_ratio=0.5,
    )
    _write_summary(
        str(parent / "rep1" / "summary.json"),
        ttft_ms_p50=20.0,
        output_tokens_per_s=200.0,
        cache_hit_ratio=0.7,
    )
    avg, n = _read_arm(str(parent))
    assert n == 2
    assert avg["ttft_ms_p50"] == pytest.approx(15.0)
    assert avg["output_tokens_per_s"] == pytest.approx(150.0)
    assert avg["cache_hit_ratio"] == pytest.approx(0.6)


def test_read_arm_returns_empty_when_no_summary(tmp_path):
    _, n = _read_arm(str(tmp_path / "does-not-exist"))
    assert n == 0
