#!/usr/bin/env python3
"""Aggregate the three-regime HiMA matrix (base vs sys, N reps).

Scans <matrix_root>/<regime>/<arm>/<arm>_r<i>.json produced by
run_arm_ts.sh / run_regime_matrix.sh and prints per regime x arm:

  * reps found / reps valid (a rep is VALID only if n_error/n_requests <= 5%
    and n_ok > 0 -- a crashed server yields all-error garbage that must not
    enter the stats)
  * throughput tok/s: mean +- std over valid reps
  * TTFT ms p50 / p90 / p99: mean +- std over valid reps
  * cache_hit mean, total n_error

Exit code 1 if any regime/arm has fewer than --min-valid valid reps, so CI
or the driver can detect an incomplete/broken matrix.

Usage:
  aggregate_regime_matrix.py <matrix_root> [--min-valid 3] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ERROR_RATE_VALID_MAX = 0.05


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def load_rep(path: Path) -> dict | None:
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  [warn] unreadable rep {path}: {exc}", file=sys.stderr)
        return None


def rep_valid(d: dict) -> tuple[bool, str]:
    n_req = d.get("n_requests") or 0
    n_ok = d.get("n_ok") or 0
    n_err = d.get("n_error") or 0
    if n_ok == 0:
        return False, "n_ok=0 (server dead for whole rep)"
    if n_req and n_err / n_req > ERROR_RATE_VALID_MAX:
        return False, f"error rate {n_err}/{n_req} > {ERROR_RATE_VALID_MAX:.0%} (mid-rep crash)"
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("matrix_root", type=Path)
    ap.add_argument("--min-valid", type=int, default=3)
    ap.add_argument("--json", type=Path, default=None, help="also dump summary JSON here")
    args = ap.parse_args()

    root: Path = args.matrix_root
    if not root.is_dir():
        print(f"matrix root {root} does not exist", file=sys.stderr)
        return 1

    summary: dict[str, dict[str, dict]] = {}
    incomplete = []

    regimes = sorted(p for p in root.iterdir() if p.is_dir())
    for regime_dir in regimes:
        regime = regime_dir.name
        arms = sorted(p for p in regime_dir.iterdir() if p.is_dir())
        if not arms:
            continue
        summary[regime] = {}
        for arm_dir in arms:
            arm = arm_dir.name
            reps = sorted(arm_dir.glob(f"{arm}_r*.json"))
            valid, invalid_notes = [], []
            for rp in reps:
                d = load_rep(rp)
                if d is None:
                    invalid_notes.append(f"{rp.name}: unreadable")
                    continue
                ok, why = rep_valid(d)
                if ok:
                    valid.append(d)
                else:
                    invalid_notes.append(f"{rp.name}: {why}")

            tput_m, tput_s = mean_std([d["throughput_tok_s"] for d in valid])
            stats = {
                "reps_found": len(reps),
                "reps_valid": len(valid),
                "invalid": invalid_notes,
                "throughput_tok_s": {"mean": tput_m, "std": tput_s},
                "cache_hit_mean": mean_std([d.get("cache_hit", float("nan")) for d in valid])[0],
                "n_error_total": sum(d.get("n_error", 0) for d in valid),
            }
            for pct in ("p50", "p90", "p99"):
                vals = [d["ttft_ms"][pct] for d in valid if isinstance(d.get("ttft_ms"), dict)]
                m, s = mean_std(vals)
                stats[f"ttft_{pct}_ms"] = {"mean": m, "std": s}
            summary[regime][arm] = stats
            if len(valid) < args.min_valid:
                incomplete.append(f"{regime}/{arm}: {len(valid)}/{args.min_valid} valid reps")

    hdr = (
        f"{'regime':<14}{'arm':<6}{'reps':>5}  {'tok/s (mean+-std)':>20}  "
        f"{'TTFT p50 ms':>16}  {'TTFT p90 ms':>18}  {'TTFT p99 ms':>18}  {'hit':>6}  {'errs':>5}"
    )
    print(hdr)
    print("-" * len(hdr))
    for regime, arms in summary.items():
        for arm, s in arms.items():
            t = s["throughput_tok_s"]
            p50, p90, p99 = (s["ttft_p50_ms"], s["ttft_p90_ms"], s["ttft_p99_ms"])
            print(
                f"{regime:<14}{arm:<6}{s['reps_valid']}/{s['reps_found']:<3}  "
                f"{t['mean']:>10.1f} +- {t['std']:<6.1f}  "
                f"{p50['mean']:>8.0f} +- {p50['std']:<5.0f}  "
                f"{p90['mean']:>9.0f} +- {p90['std']:<6.0f}  "
                f"{p99['mean']:>9.0f} +- {p99['std']:<6.0f}  "
                f"{s['cache_hit_mean']:>6.3f}  {s['n_error_total']:>5}"
            )
            for note in s["invalid"]:
                print(f"    !! invalid rep: {note}")

    if args.json:
        args.json.write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {args.json}")

    if incomplete:
        print("\nINCOMPLETE:")
        for line in incomplete:
            print(f"  {line}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
