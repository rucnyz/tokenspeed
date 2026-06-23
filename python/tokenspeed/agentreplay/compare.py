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

"""A/B comparison utility.

Usage::

    python -m tokenspeed.agentreplay.compare runs/base runs/sys

Reads one or more per-run output directories produced by
``python -m tokenspeed.agentreplay`` (each must contain a ``summary.json``)
and prints a side-by-side table with absolute + relative deltas. Multiple
reps of the same arm can be averaged by passing a directory whose direct
children are themselves run dirs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections.abc import Iterable


_METRICS_ORDER: list[tuple[str, str, str]] = [
    # (key, display name, format)
    ("ttft_ms_p50", "TTFT p50 (ms)", "{:.1f}"),
    ("ttft_ms_p90", "TTFT p90 (ms)", "{:.1f}"),
    ("ttft_ms_p99", "TTFT p99 (ms)", "{:.1f}"),
    ("ttft_ms_mean", "TTFT mean (ms)", "{:.1f}"),
    ("latency_ms_p50", "Latency p50 (ms)", "{:.1f}"),
    ("latency_ms_p99", "Latency p99 (ms)", "{:.1f}"),
    ("output_tokens_per_s", "Throughput (tok/s)", "{:.1f}"),
    ("cache_hit_ratio", "Prefix cache hit", "{:.3f}"),
    ("n_requests", "Requests", "{:.0f}"),
    ("n_failed", "Failed", "{:.0f}"),
    ("wall_seconds", "Wall (s)", "{:.1f}"),
]


def _find_summaries(path: str) -> list[str]:
    """Return the list of summary.json files under ``path``.

    Accepts either a leaf run dir (``runs/base/rep0/``) or a parent that
    holds multiple rep subdirs (``runs/base/`` -> ``[rep0, rep1, ...]``).
    """
    direct = os.path.join(path, "summary.json")
    if os.path.exists(direct):
        return [direct]
    if not os.path.isdir(path):
        return []
    out: list[str] = []
    for entry in sorted(os.listdir(path)):
        cand = os.path.join(path, entry, "summary.json")
        if os.path.exists(cand):
            out.append(cand)
    return out


def _average(values: Iterable[float]) -> float:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return float("nan")
    return statistics.mean(vals)


def _read_arm(path: str) -> tuple[dict, int]:
    """Load and average all summary.json files under ``path``.

    Returns (averaged_summary, n_reps). Falls back to an empty dict
    (n_reps == 0) when no summaries exist so the caller can flag it.
    """
    summaries = _find_summaries(path)
    if not summaries:
        return {}, 0
    raw: list[dict] = []
    for s in summaries:
        with open(s) as fh:
            raw.append(json.load(fh))
    keys = {k for r in raw for k, v in r.items() if isinstance(v, (int, float))}
    avg = {k: _average(r.get(k) for r in raw) for k in keys}
    return avg, len(summaries)


def _format_delta(a: float, b: float, *, lower_better: bool) -> str:
    if math.isnan(a) or math.isnan(b) or a == 0:
        return "    -"
    pct = (b - a) / a * 100.0
    arrow = ""
    if lower_better:
        arrow = "↓" if pct < 0 else "↑"
    else:
        arrow = "↑" if pct > 0 else "↓"
    return f"{arrow}{pct:+.1f}%"


def _print_table(base_name: str, base: dict, sys_name: str, sys_: dict) -> None:
    col_w = max(len(base_name), len(sys_name), 10)
    print(f"{'Metric':<22} {base_name:>{col_w}} {sys_name:>{col_w}}   Δ")
    print("-" * (22 + 2 + col_w + 1 + col_w + 6))
    for key, label, fmt in _METRICS_ORDER:
        a = base.get(key, float("nan"))
        b = sys_.get(key, float("nan"))
        # Lower is better for latencies / failures; higher is better for
        # throughput / cache hits / wall (longer wall means more decode
        # was actually processed, but in a fixed-trace replay the wall
        # is workload-fixed and the delta is informational).
        lower_better = (
            key.startswith("ttft_ms")
            or key.startswith("latency_ms")
            or key == "n_failed"
        )
        sign = "" if (math.isnan(a) or math.isnan(b)) else _format_delta(a, b, lower_better=lower_better)
        a_str = "nan" if math.isnan(a) else fmt.format(a)
        b_str = "nan" if math.isnan(b) else fmt.format(b)
        print(f"{label:<22} {a_str:>{col_w}} {b_str:>{col_w}}   {sign}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tokenspeed.agentreplay.compare")
    p.add_argument("base", help="Base arm run dir (or parent of rep subdirs).")
    p.add_argument("sys", help="System arm run dir (or parent of rep subdirs).")
    p.add_argument("--base-name", default="base")
    p.add_argument("--sys-name", default="sys")
    args = p.parse_args(argv)

    base, base_n = _read_arm(args.base)
    sys_, sys_n = _read_arm(args.sys)
    if base_n == 0:
        print(f"ERROR: no summary.json found under {args.base}", file=sys.stderr)
        return 1
    if sys_n == 0:
        print(f"ERROR: no summary.json found under {args.sys}", file=sys.stderr)
        return 1

    print(f"\nA/B comparison ({args.base_name} N={base_n}  vs  {args.sys_name} N={sys_n})")
    _print_table(args.base_name, base, args.sys_name, sys_)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
