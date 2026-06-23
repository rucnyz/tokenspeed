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

"""Unit tests for tools/calibrate_kappa.py.

These tests stub out budget.jsonl content so we can verify that the
calibration tool aggregates per-fire samples, extracts the runtime EWMA,
and surfaces a sensible recommended cost parameter.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

# Make tools/ importable without installing.
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import calibrate_kappa as ck  # noqa: E402


def _write_budget_jsonl(path: pathlib.Path, records: list[dict]) -> None:
    with path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _snap(t: float, **kwargs) -> dict:
    base = {
        "t": t,
        "queue_len": 0,
        "decoding": 0,
        "prefilling": 0,
        "retract_count": 0,
        "kv_free_pages": 1000,
        "kv_active_pages": 0,
        "kv_mapped_pages": 0,
        "kv_headroom_pages": 0,
        "mamba_mapped_chunks": 0,
        "fires_kv_to_mamba_total": 0,
        "fires_mamba_to_kv_total": 0,
        "fires_cancelled_total": 0,
        "xpool_ewma_xfer_us_per_page": 0.0,
        "xpool_last_fire_us": 0.0,
        "xpool_last_fire_pages": 0,
    }
    base.update(kwargs)
    return base


def test_extract_fires_picks_up_counter_advances(tmp_path: pathlib.Path) -> None:
    """A per-fire sample is recorded each time the committed counter advances."""
    path = tmp_path / "budget.jsonl"
    _write_budget_jsonl(
        path,
        [
            _snap(0.0),
            _snap(
                0.1,
                fires_mamba_to_kv_total=1,
                xpool_last_fire_us=6400.0,
                xpool_last_fire_pages=64,
                xpool_ewma_xfer_us_per_page=100.0,
            ),
            _snap(
                0.2,
                fires_mamba_to_kv_total=1,
                xpool_last_fire_us=6400.0,
                xpool_last_fire_pages=64,
                xpool_ewma_xfer_us_per_page=100.0,
            ),
            _snap(
                0.3,
                fires_mamba_to_kv_total=1,
                fires_kv_to_mamba_total=1,
                xpool_last_fire_us=4800.0,
                xpool_last_fire_pages=64,
                xpool_ewma_xfer_us_per_page=87.5,
            ),
        ],
    )
    records = list(ck._iter_budget_records(path))
    samples, ewma, snaps = ck._extract_fires(records)
    assert len(samples) == 2
    assert samples[0].pages == 64
    assert samples[0].elapsed_us == pytest.approx(6400.0)
    assert samples[0].per_page == pytest.approx(100.0)
    assert samples[1].per_page == pytest.approx(75.0)
    assert ewma == pytest.approx(87.5)
    assert len(snaps) == 4


def test_calibrate_recommends_median_with_enough_samples(tmp_path: pathlib.Path) -> None:
    """With >=16 samples the median is the recommended kappa value."""
    records: list[dict] = [_snap(0.0)]
    fires_total = 0
    us_samples = [60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0]
    for i in range(20):
        fires_total += 1
        elapsed = us_samples[i % len(us_samples)] * 64.0
        records.append(
            _snap(
                0.1 * (i + 1),
                fires_mamba_to_kv_total=fires_total,
                xpool_last_fire_us=elapsed,
                xpool_last_fire_pages=64,
                xpool_ewma_xfer_us_per_page=100.0,
            )
        )
    path = tmp_path / "budget.jsonl"
    _write_budget_jsonl(path, records)
    report = ck.calibrate([path])
    assert report["samples"]["n"] == 20
    rec = report["recommended_xpool_xfer_us_per_page"]
    assert rec is not None
    # samples sorted are 60..130 with each value appearing 2-3 times;
    # the median lies between 80 and 100 depending on how the cycle
    # rounds, so we accept the whole interior range.
    assert 80.0 <= rec <= 110.0
    assert "median" in report["recommended_rationale"]


def test_calibrate_falls_back_to_ewma_when_few_samples(tmp_path: pathlib.Path) -> None:
    """Short traces with <16 fires fall back to the runtime EWMA mean."""
    records: list[dict] = [_snap(0.0)]
    records.append(
        _snap(
            0.1,
            fires_mamba_to_kv_total=1,
            xpool_last_fire_us=12800.0,
            xpool_last_fire_pages=64,
            xpool_ewma_xfer_us_per_page=200.0,
        )
    )
    path = tmp_path / "budget.jsonl"
    _write_budget_jsonl(path, records)
    report = ck.calibrate([path])
    assert report["samples"]["n"] == 1
    rec = report["recommended_xpool_xfer_us_per_page"]
    assert rec == pytest.approx(200.0)
    assert "EWMA" in report["recommended_rationale"]


def test_calibrate_returns_none_for_empty_trace(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "budget.jsonl"
    _write_budget_jsonl(path, [_snap(0.0), _snap(0.1), _snap(0.2)])
    report = ck.calibrate([path])
    assert report["recommended_xpool_xfer_us_per_page"] is None
    assert report["samples"]["n"] == 0


def test_queue_wait_lower_bound_accumulates_time(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "budget.jsonl"
    _write_budget_jsonl(
        path,
        [
            _snap(0.0, queue_len=0),
            _snap(1.0, queue_len=5),
            _snap(2.0, queue_len=5),
            _snap(3.0, queue_len=0),
        ],
    )
    report = ck.calibrate([path])
    # Queue >0 between t=1.0 and t=3.0 => 2.0 seconds * 1e6 = 2_000_000 us
    assert report["queue_wait_lower_bound_us"] == pytest.approx(2_000_000.0)


def test_main_json_output(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
    path = tmp_path / "budget.jsonl"
    _write_budget_jsonl(
        path,
        [
            _snap(0.0),
            _snap(
                0.1,
                fires_mamba_to_kv_total=1,
                xpool_last_fire_us=6400.0,
                xpool_last_fire_pages=64,
                xpool_ewma_xfer_us_per_page=100.0,
            ),
        ],
    )
    rc = ck.main([str(path), "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["recommended_xpool_xfer_us_per_page"] == pytest.approx(100.0)
