#!/usr/bin/env python3
"""Aggregate three-arm (base/sys/sys_lru) results across reps.

Reads ``<OUT>/<arm>/rep<N>/summary.json`` for each arm and rep, prints a
per-arm mean +/- std table for the headline metrics, and flags any rep whose
run.log shows a crash (illegal memory access / traceback) or a missing summary.

Usage: aggregate_three_arm.py <OUT_DIR> <REPS>
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ARMS = ["base", "sys", "sys_lru"]
METRICS = [
    ("output_tokens_per_s", "tok/s", 2, True),
    ("ttft_ms_p90", "ttft_p90(ms)", 0, False),
    ("ttft_ms_p99", "ttft_p99(ms)", 0, False),
    ("latency_ms_p50", "lat_p50(ms)", 0, False),
    ("latency_ms_p99", "lat_p99(ms)", 0, False),
    ("cache_hit_ratio", "cache_hit", 3, True),
    ("n_finished", "finished", 0, True),
    ("n_failed", "failed", 0, False),
    ("wall_seconds", "wall_s", 0, False),
]


def _crashed(run_log: Path) -> bool:
    if not run_log.exists():
        return False
    try:
        txt = run_log.read_text(errors="ignore")
    except OSError:
        return False
    return (
        "illegal memory access" in txt
        or "CUDA error" in txt
        or "Traceback (most recent call last)" in txt
    )


def main() -> int:
    out = Path(sys.argv[1])
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    print(f"\n=== three-arm aggregate: {out} ({reps} reps) ===")
    status: dict[str, list[str]] = {}
    values: dict[str, dict[str, list[float]]] = {a: {} for a in ARMS}

    for arm in ARMS:
        status[arm] = []
        for rep in range(reps):
            d = out / arm / f"rep{rep}"
            summary = d / "summary.json"
            run_log = d / "run.log"
            if _crashed(run_log):
                status[arm].append(f"rep{rep}:CRASH")
                continue
            if not summary.exists():
                status[arm].append(f"rep{rep}:no-summary")
                continue
            status[arm].append(f"rep{rep}:ok")
            s = json.load(open(summary))
            for key, *_ in METRICS:
                if key in s and s[key] is not None:
                    values[arm].setdefault(key, []).append(float(s[key]))

    print("\nrep status:")
    for arm in ARMS:
        print(f"  {arm:8s} {' '.join(status[arm])}")

    hdr = f"\n{'metric':16s}" + "".join(f"{a:>22s}" for a in ARMS)
    print(hdr)
    print("-" * len(hdr))
    for key, label, ndig, _ in METRICS:
        row = f"{label:16s}"
        for arm in ARMS:
            vals = values[arm].get(key, [])
            if not vals:
                row += f"{'--':>22s}"
                continue
            mean = statistics.fmean(vals)
            sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            cell = f"{mean:.{ndig}f}+/-{sd:.{ndig}f}"
            row += f"{cell:>22s}"
        print(row)

    # Headline verdict: sys vs base on tok/s and latency.
    def _mean(arm: str, key: str) -> float | None:
        vals = values[arm].get(key, [])
        return statistics.fmean(vals) if vals else None

    print("\nverdict (sys vs base):")
    for key, label, ndig, higher_better in [
        ("output_tokens_per_s", "tok/s", 2, True),
        ("latency_ms_p99", "lat_p99(ms)", 0, False),
        ("ttft_ms_p90", "ttft_p90(ms)", 0, False),
    ]:
        b, s = _mean("base", key), _mean("sys", key)
        if b is None or s is None or b == 0:
            print(f"  {label:14s}: n/a")
            continue
        delta = (s - b) / b * 100.0
        win = (delta > 0) == higher_better
        print(
            f"  {label:14s}: base={b:.{ndig}f} sys={s:.{ndig}f} "
            f"({delta:+.1f}% {'WIN' if win else 'LOSS'} for sys)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
