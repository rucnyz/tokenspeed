#!/usr/bin/env python3
"""Filter a replay trace JSONL file, removing sessions likely to cause errors.

Filtering criteria (any of the following disqualifies an entire session):
  1. prompt_too_long  -- any step has len(input_ids) > --max-tokens (default 262144)
  2. empty_input      -- any step has input_ids == [] or missing
  3. zero_output      -- any step has forced_output_ids == [] or missing
  4. bad_json         -- line cannot be parsed as JSON (always skipped)

Usage:
    python tools/filter_trace.py \\
        --input  dataset/claude-code-traces/traces/cc_qwen3p5_9b.jsonl \\
        --output dataset/claude-code-traces/traces/cc_qwen3p5_9b_clean.jsonl \\
        --max-tokens 262144 \\
        --verbose
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", "-i", required=True, help="Input JSONL trace path")
    p.add_argument("--output", "-o", required=True, help="Output JSONL trace path")
    p.add_argument("--max-tokens", type=int, default=262144,
                   help="Max allowed prompt tokens per step (default: 262144)")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # --- Pass 1: collect all steps per session ---
    sessions: dict[str, list[dict]] = collections.defaultdict(list)
    bad_json_lines = 0
    with inp.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                bad_json_lines += 1
                if args.verbose:
                    print(f"  [SKIP] line {lineno}: bad JSON", file=sys.stderr)
                continue
            sessions[row["program_id"]].append(row)

    # --- Pass 2: classify each session ---
    bad_sessions: dict[str, str] = {}  # pid → reason

    for pid, steps in sessions.items():
        if not steps:
            bad_sessions[pid] = "empty_session"
            continue
        for step in steps:
            input_ids = step.get("input_ids") or []
            forced_output_ids = step.get("forced_output_ids") or []
            n_prompt = len(input_ids)

            if n_prompt == 0:
                bad_sessions[pid] = f"empty_input_step{step.get('step', '?')}"
                break
            if n_prompt > args.max_tokens:
                bad_sessions[pid] = f"prompt_too_long_{n_prompt}_step{step.get('step', '?')}"
                break
            if len(forced_output_ids) == 0:
                bad_sessions[pid] = f"zero_output_step{step.get('step', '?')}"
                break

    # --- Stats ---
    total_sessions = len(sessions)
    clean_sessions = total_sessions - len(bad_sessions)
    total_steps = sum(len(v) for v in sessions.values())
    clean_steps = sum(len(v) for pid, v in sessions.items() if pid not in bad_sessions)

    print(f"Input:          {inp}")
    print(f"Output:         {out}")
    print(f"Max tokens:     {args.max_tokens:,}")
    print()
    print(f"Total sessions: {total_sessions}")
    print(f"Bad sessions:   {len(bad_sessions)}")
    print(f"Clean sessions: {clean_sessions}")
    print(f"Total steps:    {total_steps}")
    print(f"Clean steps:    {clean_steps}")
    print(f"Bad JSON lines: {bad_json_lines}")
    print()

    if args.verbose and bad_sessions:
        print("Removed sessions:")
        for pid, reason in sorted(bad_sessions.items()):
            print(f"  {pid}  [{reason}]")
        print()

    # Break down bad reasons
    reason_counts: dict[str, int] = collections.Counter()
    for reason in bad_sessions.values():
        key = reason.split("_step")[0]
        reason_counts[key] += 1
    if reason_counts:
        print("Removal reasons:")
        for k, v in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")
        print()

    # --- Write clean trace (preserving original line order) ---
    written = 0
    with inp.open(encoding="utf-8") as fh, out.open("w", encoding="utf-8") as wh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if row["program_id"] in bad_sessions:
                continue
            wh.write(raw + "\n")
            written += 1

    print(f"Written {written} lines ({clean_sessions} sessions) → {out}")


if __name__ == "__main__":
    main()
