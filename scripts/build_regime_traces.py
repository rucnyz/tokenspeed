#!/usr/bin/env python3
"""Derive long_horizon / agent_swarm / shifting replay traces from a full
converted corpus trace, following the Table 3 methodology:

  long-horizon: the deepest sessions, keeping only their deep-context
                requests (> --deep-threshold tokens).
  agent swarm:  the sessions that spawn the most sub-agents (kept whole,
                main + all linked sub-agent turns, so the harness replays
                the real concurrent fan-out).
  shifting:     a synthetic mix of deep-context, swarm, and ordinary
                sessions (stitched back-to-back with rescaled `t`).

Input schema: agentreplay/schema.py (t, program_id, step, parent_program_id,
spawned_at_step, spawn_ts, input_ids, forced_output_ids, tool_gap_after).

Usage:
    python scripts/build_regime_traces.py \
        --input dataset/claude-code-traces/traces/cc_qwen_full_corpus_deep.jsonl \
        --out-dir dataset/claude-code-traces/traces \
        --deep-threshold 300000
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics as st
from pathlib import Path
from typing import Any, Dict, List


def load(path: Path) -> List[Dict[str, Any]]:
    recs = []
    with path.open(errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs


def write(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(records, key=lambda r: float(r.get("t", 0.0)))
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def pct(xs: List[int], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f = int(k)
    if f + 1 >= len(xs):
        return xs[f]
    return xs[f] + (xs[f + 1] - xs[f]) * (k - f)


def characterize(name: str, recs: List[Dict[str, Any]]) -> None:
    progs = {r["program_id"] for r in recs}
    subs = {r["program_id"] for r in recs if r.get("parent_program_id")}
    tlens = [len(r.get("input_ids") or []) for r in recs]
    print(f"=== {name} ===")
    print(f"  requests={len(recs)} programs={len(progs)} sub-agent-programs={len(subs)}")
    if tlens:
        print(f"  tok avg={st.mean(tlens):,.0f} p50={pct(tlens,50):,.0f} "
              f"p90={pct(tlens,90):,.0f} max={max(tlens):,}")


def build_program_index(recs: List[Dict[str, Any]]):
    by_prog: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in recs:
        by_prog[r["program_id"]].append(r)
    for pid in by_prog:
        by_prog[pid].sort(key=lambda r: int(r.get("step", 0)))
    return by_prog


def root_of(pid: str, parent_of: Dict[str, str]) -> str:
    seen = set()
    cur = pid
    while cur in parent_of and parent_of[cur] and parent_of[cur] not in seen:
        seen.add(cur)
        cur = parent_of[cur]
    return cur


def build_long_horizon(by_prog, deep_threshold: int, min_requests: int) -> List[Dict[str, Any]]:
    """Deepest sessions -> keep only their turns whose input_ids exceed
    deep_threshold. Sessions are ranked by their own max depth; we keep
    adding sessions (deepest first) until we have >= min_requests deep
    turns, mirroring the paper's "deepest sessions, deep-context requests
    only" rule while giving us enough load to replay."""
    depth_by_prog = {pid: max((len(r.get("input_ids") or []) for r in steps), default=0)
                      for pid, steps in by_prog.items()}
    ranked = sorted(depth_by_prog.items(), key=lambda kv: -kv[1])
    out: List[Dict[str, Any]] = []
    for pid, depth in ranked:
        if depth <= deep_threshold:
            break
        deep_steps = [r for r in by_prog[pid] if len(r.get("input_ids") or []) > deep_threshold]
        out.extend(deep_steps)
        if len(out) >= min_requests:
            break
    return out


def build_swarm(by_prog, parent_of, top_n_roots: int) -> List[Dict[str, Any]]:
    """Sessions that spawn the most sub-agents -> keep the WHOLE call tree
    (main + every linked sub-agent's turns) so the harness replays the real
    concurrent fan-out via spawned_at_step."""
    children_count: Dict[str, int] = collections.Counter()
    for pid, ppid in parent_of.items():
        if ppid:
            children_count[root_of(ppid, parent_of)] += 1

    ranked_roots = sorted(children_count.items(), key=lambda kv: -kv[1])[:top_n_roots]
    root_set = {pid for pid, _ in ranked_roots}

    out: List[Dict[str, Any]] = []
    for pid, steps in by_prog.items():
        if root_of(pid, parent_of) in root_set:
            out.extend(steps)
    return out


def build_shifting(long_horizon: List[Dict[str, Any]], swarm: List[Dict[str, Any]],
                    by_prog, parent_of, n_ordinary_progs: int,
                    seed: int = 0) -> List[Dict[str, Any]]:
    """Synthetic mix: deep-context slice + swarm slice + a basket of
    ordinary (shallow, no-subagent) sessions, restarted back-to-back on a
    shared synthetic timeline (t rescaled per segment)."""
    import random
    rng = random.Random(seed)

    used_roots = {root_of(r["program_id"], parent_of) for r in long_horizon}
    used_roots |= {root_of(r["program_id"], parent_of) for r in swarm}

    ordinary_roots = [pid for pid, steps in by_prog.items()
                      if root_of(pid, parent_of) not in used_roots
                      and parent_of.get(pid) is None]
    rng.shuffle(ordinary_roots)
    chosen = ordinary_roots[:n_ordinary_progs]
    ordinary: List[Dict[str, Any]] = []
    for pid in chosen:
        ordinary.extend(by_prog[pid])

    segments = [long_horizon, swarm, ordinary]
    out: List[Dict[str, Any]] = []
    t_off = 0.0
    for seg in segments:
        if not seg:
            continue
        seg_sorted = sorted(seg, key=lambda r: float(r.get("t", 0.0)))
        seg_t0 = seg_sorted[0].get("t", 0.0)
        seg_tmax = seg_sorted[-1].get("t", 0.0)
        for r in seg_sorted:
            r2 = dict(r)
            r2["t"] = round(t_off + (r.get("t", 0.0) - seg_t0), 4)
            out.append(r2)
        t_off += (seg_tmax - seg_t0) + 30.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--deep-threshold", type=int, default=300_000)
    ap.add_argument("--long-horizon-min-requests", type=int, default=300)
    ap.add_argument("--swarm-top-roots", type=int, default=5)
    ap.add_argument("--shifting-ordinary-progs", type=int, default=60)
    a = ap.parse_args()

    recs = load(Path(a.input))
    characterize("full_corpus", recs)

    by_prog = build_program_index(recs)
    parent_of = {pid: steps[0].get("parent_program_id") for pid, steps in by_prog.items()}

    long_horizon = build_long_horizon(by_prog, a.deep_threshold, a.long_horizon_min_requests)
    swarm = build_swarm(by_prog, parent_of, a.swarm_top_roots)
    shifting = build_shifting(long_horizon, swarm, by_prog, parent_of,
                               a.shifting_ordinary_progs)

    out_dir = Path(a.out_dir)
    write(long_horizon, out_dir / "cc_qwen_long_horizon_v2.jsonl")
    write(swarm, out_dir / "cc_qwen_swarm_v2.jsonl")
    write(shifting, out_dir / "cc_qwen_shifting_v2.jsonl")

    print()
    characterize("long_horizon_v2", long_horizon)
    characterize("swarm_v2", swarm)
    characterize("shifting_v2", shifting)


if __name__ == "__main__":
    main()
