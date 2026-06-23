#!/usr/bin/env python3
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

"""Calibrate XPool fire-cost from a runtime budget.jsonl trace.

The event-loop snapshotter writes one JSON object per replay metrics tick
(default 100 ms) to ``budget.jsonl``.  Each snapshot includes the running
EWMA of microseconds-per-KV-page observed by ``XPoolActuator``
(``xpool_ewma_xfer_us_per_page``).  This tool walks one or more
``budget.jsonl`` files, surfaces summary statistics over the *committed*
fires, and emits a recommended ``xpool_xfer_us_per_page`` value plus an
``xpool_queue_wait_us`` lower bound derived from queue/retract pressure.

Usage
-----
::

    python tools/calibrate_kappa.py runs/.../budget.jsonl [more ...]
    python tools/calibrate_kappa.py --recursive runs/stress40_v1
    python tools/calibrate_kappa.py runs/.../budget.jsonl --emit-overrides

``--emit-overrides`` prints CLI overrides ready to paste into a
``--override key=value`` invocation of ``tokenspeed.agentreplay``.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass


@dataclass(slots=True)
class FireSample:
    elapsed_us: float
    pages: int

    @property
    def per_page(self) -> float:
        return self.elapsed_us / float(self.pages) if self.pages > 0 else 0.0


def _iter_budget_records(path: pathlib.Path) -> Iterator[dict]:
    """Yield JSON snapshots from a ``budget.jsonl`` file."""
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _extract_fires(records: Iterable[dict]) -> tuple[list[FireSample], float, list[dict]]:
    """Return all committed-fire samples and the final EWMA seen in the trace.

    Snapshots include the cumulative committed-fire counter and the latest
    fire's elapsed_us / pages.  We sample the per-fire pair whenever the
    counter advances; the EWMA is a side-product of the actuator and we
    just record its final value (useful when the trace is very short).
    """
    samples: list[FireSample] = []
    last_total = 0
    last_ewma = 0.0
    queue_snapshots: list[dict] = []
    for rec in records:
        ewma = float(rec.get("xpool_ewma_xfer_us_per_page", 0.0))
        if ewma > 0.0:
            last_ewma = ewma
        total = int(rec.get("fires_kv_to_mamba_total", 0)) + int(
            rec.get("fires_mamba_to_kv_total", 0)
        )
        if total > last_total:
            pages = int(rec.get("xpool_last_fire_pages", 0))
            elapsed_us = float(rec.get("xpool_last_fire_us", 0.0))
            if pages > 0 and elapsed_us > 0.0:
                # The snapshot interval may skip individual fires when the
                # actuator commits >1 fire between ticks.  In that case we
                # still see only the latest sample; this is best-effort
                # (sampling bias toward longer fires), which matches how
                # the EWMA itself sees the data.
                samples.append(FireSample(elapsed_us=elapsed_us, pages=pages))
            last_total = total
        queue_snapshots.append(rec)
    return samples, last_ewma, queue_snapshots


def _pct(samples: list[float], p: float) -> float:
    """Return the p-th percentile (0-100) using nearest-rank."""
    if not samples:
        return 0.0
    s = sorted(samples)
    k = max(0, min(len(s) - 1, int(math.ceil(p / 100.0 * len(s))) - 1))
    return s[k]


def _summarise(per_page: list[float]) -> dict:
    if not per_page:
        return {"n": 0}
    return {
        "n": len(per_page),
        "min": min(per_page),
        "max": max(per_page),
        "mean": statistics.fmean(per_page),
        "median": statistics.median(per_page),
        "p90": _pct(per_page, 90.0),
        "p99": _pct(per_page, 99.0),
    }


def _queue_wait_lower_bound(queue_snapshots: list[dict]) -> float:
    """Estimate ``xpool_queue_wait_us`` from the longest queueing event.

    We treat any tick with ``queue_len`` > 0 as evidence of queueing
    pressure and convert the cumulative time spent above-zero into a
    coarse wait time in microseconds.  This is *only* a lower bound: it
    counts the time pressure was visible, not the per-request wait.  The
    user should fold the engine-side TTFT analysis in for a real upper
    bound.  Returning the lower bound is still useful because it tells
    the budgeter "queue waits are at least this expensive" so it stops
    under-weighting the queue signal in cost comparisons.
    """
    if len(queue_snapshots) < 2:
        return 0.0
    total_us = 0.0
    for i in range(1, len(queue_snapshots)):
        prev = queue_snapshots[i - 1]
        cur = queue_snapshots[i]
        if int(prev.get("queue_len", 0)) > 0:
            dt = float(cur.get("t", 0.0)) - float(prev.get("t", 0.0))
            if dt > 0.0:
                total_us += dt * 1_000_000.0
    return total_us


def _discover_paths(args: argparse.Namespace) -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for p in args.paths:
        path = pathlib.Path(p)
        if path.is_dir():
            if args.recursive:
                out.extend(sorted(path.rglob("budget.jsonl")))
            else:
                candidate = path / "budget.jsonl"
                if candidate.is_file():
                    out.append(candidate)
        elif path.is_file():
            out.append(path)
        else:
            print(f"warning: not a file or directory: {p}", file=sys.stderr)
    return out


def calibrate(paths: list[pathlib.Path]) -> dict:
    """Aggregate budget.jsonl traces into a single calibration report."""
    all_samples: list[FireSample] = []
    ewmas: list[float] = []
    queue_us_total = 0.0
    per_file_reports: list[dict] = []
    for path in paths:
        records = list(_iter_budget_records(path))
        samples, ewma, snaps = _extract_fires(records)
        all_samples.extend(samples)
        if ewma > 0.0:
            ewmas.append(ewma)
        q_us = _queue_wait_lower_bound(snaps)
        queue_us_total += q_us
        per_file_reports.append(
            {
                "file": str(path),
                "n_snapshots": len(records),
                "n_fires": len(samples),
                "final_ewma_us_per_page": ewma,
                "queue_wait_lower_bound_us": q_us,
            }
        )

    per_page = [s.per_page for s in all_samples]
    per_page_stats = _summarise(per_page)
    # Recommended kappa: prefer the median if we have at least 16 samples,
    # else fall back to the average of the per-file final EWMAs.  This
    # tends to stay close to typical fire cost while resisting the long
    # tail from rare large transfers.
    if per_page_stats["n"] >= 16:
        recommended = per_page_stats["median"]
        rationale = "median over per-fire samples"
    elif ewmas:
        recommended = statistics.fmean(ewmas)
        rationale = "mean over per-file final EWMA (insufficient fires for median)"
    elif per_page_stats["n"] > 0:
        recommended = per_page_stats["mean"]
        rationale = "mean over per-fire samples (small sample fallback)"
    else:
        recommended = None
        rationale = "no samples"
    return {
        "files": per_file_reports,
        "samples": per_page_stats,
        "final_ewma_us_per_page_mean": (
            statistics.fmean(ewmas) if ewmas else None
        ),
        "queue_wait_lower_bound_us": queue_us_total,
        "recommended_xpool_xfer_us_per_page": recommended,
        "recommended_rationale": rationale,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="calibrate_kappa")
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more budget.jsonl files (or directories containing them).",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="When a path is a directory, descend recursively looking for budget.jsonl.",
    )
    parser.add_argument(
        "--emit-overrides",
        action="store_true",
        help="Print '--override key=value' fragments ready for tokenspeed.agentreplay.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full report as JSON instead of the human-readable summary.",
    )
    args = parser.parse_args(argv)

    paths = _discover_paths(args)
    if not paths:
        print("error: no budget.jsonl files found", file=sys.stderr)
        return 2

    report = calibrate(paths)
    if args.json:
        json.dump(report, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0

    print(f"Scanned {len(paths)} budget.jsonl trace(s):")
    for entry in report["files"]:
        print(
            f"  {entry['file']}: snapshots={entry['n_snapshots']} "
            f"fires={entry['n_fires']} ewma={entry['final_ewma_us_per_page']:.2f} us/page"
        )
    s = report["samples"]
    if s["n"] > 0:
        print(
            "\nPer-fire μs/page: "
            f"n={s['n']} min={s['min']:.1f} median={s['median']:.1f} "
            f"mean={s['mean']:.1f} p90={s['p90']:.1f} p99={s['p99']:.1f} max={s['max']:.1f}"
        )
    else:
        print("\nPer-fire μs/page: no samples")
    if report["final_ewma_us_per_page_mean"] is not None:
        print(
            f"Final runtime EWMA (mean over files): "
            f"{report['final_ewma_us_per_page_mean']:.2f} us/page"
        )
    print(
        f"Queue-wait lower bound (sum): "
        f"{report['queue_wait_lower_bound_us']:.0f} us"
    )

    rec = report["recommended_xpool_xfer_us_per_page"]
    if rec is None:
        print("\nNo recommendation — collect more fires (>0 committed) and retry.")
    else:
        print(
            f"\nRecommended xpool_xfer_us_per_page = {rec:.2f}  "
            f"({report['recommended_rationale']})"
        )
        if args.emit_overrides:
            print(
                "\nOverride fragment:\n"
                f"  --override xpool_xfer_us_per_page={rec:.2f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
