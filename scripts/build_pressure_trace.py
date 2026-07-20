#!/usr/bin/env python3
"""Build a synthetic sustained-pressure trace from a cc_qwen*.jsonl trace.

Motivation (see ``docs/guides/hima_phase3.md`` for the measured decile
breakdown): replaying the raw trace with ``--max-sessions N`` and the
replayer's default ``--max-inter-session-gap-s`` collapses nearly all
session arrivals into the first few minutes of wall time (because the cap
squashes the real multi-day gaps between sessions down to a few seconds
each). Concurrency -- and therefore KV pressure -- then decays to near-zero
for most of the run instead of staying high throughout.

This dataset's sessions form a two-level tree (root sessions, some of which
spawn subagent sessions via ``parent_program_id``/``spawn_ts``). Naively
replaying a root together with its subagents as a single time-shifted
"cluster" breaks down for this trace: several roots reuse the same
``program_id`` across real usage spanning *weeks*, so a root's subagents can
have ``spawn_ts`` values days to a month apart from the root's own steps and
from each other. Preserving that relative timing while shifting the whole
cluster produces an absurd multi-week-long synthetic session.

Instead, this script treats every session (root or subagent) as an
independent replay unit -- only ``tool_gap_after`` (intra-session step
pacing) is trusted from the original data; every session's *arrival* time is
picked by this script, not carried over from the original ``t``/``spawn_ts``.
Sessions are split by total prompt-token footprint into:

* "long" sessions (top ``--long-top-n`` by footprint) -- repeated once per
  generation, staggered within each generation, to sustain a KV-heavy
  concurrent-decode floor for the whole run.
* "short" sessions (the rest) -- a rotating slice of ``--short-burst-size``
  of them is crammed into a narrow ``--short-burst-window-s`` window once
  per generation, simulating a burst of many new sessions arriving at once
  (Mamba/prefill-heavy), interleaved with the sustained long-session load.

Rows are streamed straight to ``--out`` (not buffered in memory) because
this trace's ``input_ids`` lists get very large for deep multi-turn chains.

Output is consumed the same way as any other cc_qwen*.jsonl trace, e.g.:

    python -u -m tokenspeed.agentreplay --trace <out> --time-scale 0.35 \\
        --max-inter-session-gap-s -1 ...

Passing a negative/zero ``--max-inter-session-gap-s`` to agentreplay is
important -- otherwise the replayer's own gap-capping logic will re-collapse
the arrival schedule built here (which is spread out on purpose).
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def _load_raw(path: str) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _anchor_field(first_row: dict) -> str:
    return "spawn_ts" if first_row.get("spawn_ts") is not None else "t"


def build(args: argparse.Namespace, out_fh) -> tuple[int, int, float]:
    rows = _load_raw(args.trace)
    by_pid: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_pid[r["program_id"]].append(r)
    for pid in by_pid:
        by_pid[pid].sort(key=lambda r: r["step"])

    def footprint(pid: str) -> int:
        return sum(len(r["input_ids"]) for r in by_pid[pid])

    pids_sorted = sorted(by_pid, key=footprint, reverse=True)
    long_pids = pids_sorted[: args.long_top_n]
    short_pool = pids_sorted[args.long_top_n :]

    print(
        f"[build_pressure_trace] {len(pids_sorted)} sessions total; "
        f"{len(long_pids)} long (footprint sum="
        f"{sum(footprint(p) for p in long_pids)}), "
        f"{len(short_pool)} short-pool (mean footprint="
        f"{sum(footprint(p) for p in short_pool) / max(len(short_pool), 1):.0f})"
    )

    n_rows = 0
    n_sessions = 0
    t_min = float("inf")
    t_max = float("-inf")

    def emit_session(pid: str, tag: str, target_start: float) -> None:
        nonlocal n_rows, n_sessions, t_min, t_max
        steps = by_pid[pid]
        anchor = _anchor_field(steps[0])
        shift = target_start - float(steps[0][anchor])
        new_pid = f"{pid}__{tag}"
        n_sessions += 1
        for r in steps:
            nr = dict(r)
            nr["program_id"] = new_pid
            # Subagent parent linkage is metadata-only here (dispatch timing
            # no longer derives from it), but keep it internally consistent
            # for any downstream tooling that groups by parent_program_id.
            if r["parent_program_id"] is not None:
                nr["parent_program_id"] = f"{r['parent_program_id']}__{tag}"
            nr["t"] = r["t"] + shift
            if r["spawn_ts"] is not None:
                nr["spawn_ts"] = r["spawn_ts"] + shift
            out_fh.write(json.dumps(nr) + "\n")
            n_rows += 1
        # Only the first row's anchor value drives dispatch (see module
        # docstring); measure horizon from that alone so it reflects arrival
        # spread, not a session's own internal step-to-step ``t`` drift.
        first_anchor_val = float(steps[0][anchor]) + shift
        t_min = min(t_min, first_anchor_val)
        t_max = max(t_max, first_anchor_val)

    for g in range(args.generations):
        gen_start = g * args.stagger_s

        long_spread = args.long_spread_s if args.long_spread_s > 0 else 1.0
        for i, pid in enumerate(long_pids):
            target = gen_start + (i / max(len(long_pids) - 1, 1)) * long_spread
            emit_session(pid, f"g{g}_long{i}", target)

        n_burst = args.short_burst_size
        for j in range(n_burst):
            pid = short_pool[(g * n_burst + j) % len(short_pool)]
            offset = (j / max(n_burst - 1, 1)) * args.short_burst_window_s
            target = gen_start + args.short_phase_delay_s + offset
            emit_session(pid, f"g{g}_short{j}", target)

    horizon = t_max - t_min
    return n_rows, n_sessions, horizon


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--trace", required=True, help="Source cc_qwen*.jsonl trace.")
    ap.add_argument("--out", required=True, help="Output jsonl path.")
    ap.add_argument(
        "--generations",
        type=int,
        default=4,
        help="How many staggered repeats of the long/short session sets to emit.",
    )
    ap.add_argument(
        "--stagger-s",
        type=float,
        default=1200.0,
        help="Wall-clock-target offset (pre time-scale) between generation starts.",
    )
    ap.add_argument(
        "--long-top-n",
        type=int,
        default=20,
        help="Number of highest-footprint sessions treated as sustained "
        "KV-heavy load; repeated once per generation.",
    )
    ap.add_argument(
        "--long-spread-s",
        type=float,
        default=60.0,
        help="Window within a generation that the long sessions' start "
        "times are spread across (avoids all starting at the same instant).",
    )
    ap.add_argument(
        "--short-burst-size",
        type=int,
        default=30,
        help="Number of short sessions injected per generation's burst "
        "window (a rotating slice of the short-footprint pool; can exceed "
        "pool size, in which case sessions repeat with a fresh copy).",
    )
    ap.add_argument(
        "--short-burst-window-s",
        type=float,
        default=200.0,
        help="Wall-clock-target window the short-burst copies are spread across.",
    )
    ap.add_argument(
        "--short-phase-delay-s",
        type=float,
        default=400.0,
        help="Wall-clock-target delay from generation start to the short-burst phase.",
    )
    ap.add_argument(
        "--time-scale-hint",
        type=float,
        default=0.35,
        help="Only used to print the expected wall-clock horizon; does not "
        "affect the emitted trace (pass the real value to agentreplay).",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        n_rows, n_sessions, horizon = build(args, fh)

    print(
        f"[build_pressure_trace] wrote {n_rows} steps across {n_sessions} "
        f"sessions to {out_path}; target arrival horizon={horizon:.0f}s "
        f"(~{horizon / max(args.time_scale_hint, 1e-9):.0f}s wall at "
        f"time_scale={args.time_scale_hint})"
    )


if __name__ == "__main__":
    main()
