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

"""CLI entry: ``python -m tokenspeed.agentreplay ...``

Drives a single replay run with one of two presets (``base`` or ``sys``)
plus arbitrary engine knob overrides. The output directory contains:

* ``per_request.jsonl`` -- one record per replayed step
* ``summary.json``      -- aggregated TTFT / throughput / cache-hit
* ``config.json``       -- the exact config used (for reproducibility)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time

from tokenspeed.agentreplay.metrics import aggregate, write_jsonl
from tokenspeed.agentreplay.replayer import ReplayConfig, run_replay
from tokenspeed.runtime.entrypoints.engine import Engine

logger = logging.getLogger("tokenspeed.agentreplay")


# Sensible HiMA defaults for the two arms. Anything explicitly passed
# via --override wins over these. Keeping them here (not on the CLI)
# avoids reproducibility drift between runs and gives us one obvious
# place to update when defaults change.
_PRESETS: dict[str, dict] = {
    "base": {
        "radix_eviction_policy": "lru",
        "enable_budgeter": False,
        "enable_admitter": False,
        "enable_xpool_dynamic_capacity": False,
    },
    "sys": {
        # Canonical HiMA decision layer: LPB eviction + budgeter +
        # admitter + XPool dynamic capacity (logical-only Mamba path).
        #
        # S2.2-followup (Mamba tensor rebind): the Mamba pool is now
        # sized to base + xpool_mamba_headroom_slots at boot, the arena
        # is fully pre-mapped, and the actuator skips physical handle
        # transfers on the Mamba side (the conv/ssm contiguous-slice
        # layout cannot tolerate post-boot resize).  kv_to_mamba and
        # mamba_to_kv fires still update the C++ allocator's slot bound
        # and drive KV-side physical re-mapping; Mamba stays static
        # under the hood, so stress runs (>=30 concurrent decoders) no
        # longer crash inside fused_sigmoid_gating_delta_rule_update.
        #
        # NOTE on LPB vs LRU: on the cc_qwen_mamba trace (20 sessions, ~3
        # steps each, sparse cross-session prefix reuse) LPB ranks below
        # LRU on cache_hit because each prefix only sees 1-2 hits in its
        # lifetime, so LPB's "evict cold first" rule preempts nodes that
        # LRU would have kept on recency alone. Use the `sys_lru` preset
        # or `--override radix_eviction_policy=lru` for an LRU+HiMA
        # ablation.
        "radix_eviction_policy": "lpb",
        "lpb_window_s": 60.0,
        "lpb_hit_deque_maxlen": 4096,
        "csigma_kv_alpha": 1.02e-7,
        "csigma_kv_beta": 0.0246,
        "csigma_kv_gamma": 5.97,
        "csigma_m": 0.0,
        "enable_budgeter": True,
        "enable_admitter": True,
        "budgeter_tick_s": 1.0,
        "budgeter_pages_per_fire": 64,
        "enable_xpool_dynamic_capacity": True,
        # S2.1: skip XPool fire-decision and per-request admit gating when
        # both pools are below 50% util. Pass --override
        # xpool_saturation_low=0.0 to A/B against the legacy (always-on)
        # behavior on the same trace.
        "xpool_saturation_low": 0.5,
        # S2.2: after a fire commits in direction D, suppress fires in the
        # opposite direction for this many seconds. Prevents direction
        # thrashing under oscillating pressure. Same-direction fires are
        # not gated. Override to 0.0 to disable.
        "xpool_reverse_cooldown_s": 2.0,
        # S2.2-followup: extra Mamba slots the C++ allocator can grow
        # into.  0 = auto-derive (max(pages_per_fire//4, 32)).  Larger
        # values let mamba_to_kv→kv_to_mamba cycles run longer before
        # hitting the slot ceiling, at the cost of permanently reserved
        # GPU memory (~mamba_bytes_per_slot per extra slot).
        "xpool_mamba_headroom_slots": 0,
        # S2.3: PressureAdapter blends EWMA(queue_len), EWMA(retracted),
        # EWMA(paused) into the budgeter's direction-decision pressure.
        # Default weights are conservative: a fully saturated queue lifts
        # adjusted KV pressure by 0.25, a full retract burst by another
        # 0.20.  Paused signal is reserved for a future admitter defer
        # counter and stays 0 until that lands.  Reference counts of 0
        # auto-derive (queue_ref=max_num_seqs/2, retract_ref=max_num_seqs/4).
        "xpool_w_queue": 0.25,
        "xpool_w_retract": 0.20,
        "xpool_w_paused": 0.0,
        "xpool_queue_ref": 0,
        "xpool_retract_ref": 0,
        "xpool_paused_ref": 0,
        # S2.7: clamp max_batch_size to mamba_total_slots per tick so we
        # stop admitting decoders that cannot fit in Mamba after a
        # kv_to_mamba transfer.  Cheap insurance — no-op when no fires
        # happen — that prevents post-fire admit-then-retract churn.
        "enable_dynamic_admission_cap": True,
    },
    # Ablation preset: same HiMA decision layer as `sys`, but with the
    # baseline LRU eviction. Useful for isolating LPB-vs-LRU effects when
    # comparing to either the `base` preset (LRU, no HiMA decision) or the
    # canonical `sys` preset (LPB + HiMA decision).
    "sys_lru": {
        "radix_eviction_policy": "lru",
        "enable_budgeter": True,
        "enable_admitter": True,
        "budgeter_tick_s": 1.0,
        "budgeter_pages_per_fire": 64,
        "enable_xpool_dynamic_capacity": True,
        "xpool_saturation_low": 0.5,
        "xpool_reverse_cooldown_s": 2.0,
        "xpool_mamba_headroom_slots": 0,
        "xpool_w_queue": 0.25,
        "xpool_w_retract": 0.20,
        "xpool_w_paused": 0.0,
        "xpool_queue_ref": 0,
        "xpool_retract_ref": 0,
        "xpool_paused_ref": 0,
        "enable_dynamic_admission_cap": True,
    },
}


def _parse_overrides(overrides: list[str]) -> dict:
    """Parse ``--override key=value`` strings into a kwargs dict.

    Values are passed through ``json.loads`` so ``true``/``false``,
    integers, and floats get the right type. Bare strings fall through
    unchanged. This lets the same flag drive ``budgeter_pages_per_fire=64``
    and ``radix_eviction_policy=lpb`` without per-key parsing logic.
    """
    out: dict = {}
    for kv in overrides:
        if "=" not in kv:
            raise SystemExit(f"--override must be key=value, got {kv!r}")
        k, _, v = kv.partition("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def _build_engine_kwargs(args: argparse.Namespace) -> dict:
    kwargs: dict = {
        "model": args.model,
        "dtype": args.dtype,
        "attention_backend": args.attention_backend,
        "mamba_full_memory_ratio": args.mamba_full_memory_ratio,
    }
    if args.base_gpu_id is not None:
        kwargs["base_gpu_id"] = args.base_gpu_id
    kwargs.update(_PRESETS[args.preset])
    kwargs.update(_parse_overrides(args.override))
    return kwargs


def _maybe_int(v: str | None) -> int | None:
    return None if v is None else int(v)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tokenspeed.agentreplay")
    p.add_argument("--trace", required=True, help="Path to a cc_qwen*.jsonl trace file.")
    p.add_argument("--model", required=True, help="HuggingFace path to the model to serve.")
    p.add_argument("--output-dir", required=True, help="Directory to write metrics into.")
    p.add_argument(
        "--preset",
        choices=sorted(_PRESETS),
        default="base",
        help="HiMA preset: 'base' (LRU, no budgeter) or 'sys' (LPB + budgeter + admitter).",
    )
    p.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="key=value",
        help="Override a single Engine kwarg (repeatable). Values are JSON-decoded.",
    )
    # Workload shape
    p.add_argument("--max-sessions", type=_maybe_int, default=None)
    p.add_argument("--time-scale", type=float, default=1.0)
    p.add_argument("--warmup-seconds", type=float, default=0.0)
    p.add_argument("--request-timeout-s", type=float, default=600.0)
    p.add_argument(
        "--no-normalize-time",
        dest="normalize_time",
        action="store_false",
        help=(
            "Don't shift trace timestamps so the first session lands at t=0. "
            "Default is to normalize because cc_qwen* traces store absolute "
            "wall-clock timestamps spanning weeks."
        ),
    )
    p.set_defaults(normalize_time=True)
    p.add_argument(
        "--max-inter-session-gap-s",
        type=float,
        default=5.0,
        help=(
            "Cap the wall-clock gap between consecutive sessions' first steps "
            "to this many seconds (default 5.0). Set <= 0 to disable. Does "
            "not affect intra-session tool_gap_after delays."
        ),
    )
    # Engine boot
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attention-backend", default="triton")
    p.add_argument("--mamba-full-memory-ratio", type=float, default=0.15)
    p.add_argument("--base-gpu-id", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_argparser().parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    engine_kwargs = _build_engine_kwargs(args)
    max_gap = args.max_inter_session_gap_s
    if max_gap is not None and max_gap <= 0:
        max_gap = None
    cfg = ReplayConfig(
        trace_path=args.trace,
        time_scale=args.time_scale,
        max_sessions=args.max_sessions,
        warmup_seconds=args.warmup_seconds,
        request_timeout_s=args.request_timeout_s,
        normalize_time=args.normalize_time,
        max_inter_session_gap_s=max_gap,
    )

    config_dump = {
        "preset": args.preset,
        "trace": args.trace,
        "model": args.model,
        "max_sessions": args.max_sessions,
        "time_scale": args.time_scale,
        "warmup_seconds": args.warmup_seconds,
        "normalize_time": args.normalize_time,
        "max_inter_session_gap_s": max_gap,
        "engine_kwargs": engine_kwargs,
        "start_wall_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as fh:
        json.dump(config_dump, fh, indent=2, default=str)

    # Have the scheduler dump its tick-level pool state next to the per-request
    # log. The event loop honors this env var even though the scheduler runs in
    # a subprocess: the value is inherited because we spawn before fork (see
    # entrypoints.engine.Engine.__init__).
    metrics_path = os.path.join(args.output_dir, "budget.jsonl")
    os.environ["TOKENSPEED_REPLAY_METRICS_PATH"] = metrics_path
    os.environ.setdefault("TOKENSPEED_REPLAY_METRICS_INTERVAL_S", "0.1")

    logger.info("booting engine with preset=%s", args.preset)
    engine = Engine(**engine_kwargs)
    try:
        metrics, wall_start, wall_end = asyncio.run(run_replay(engine, cfg))
    finally:
        try:
            engine.shutdown()
        except Exception:  # pragma: no cover - best-effort teardown
            logger.exception("engine.shutdown raised; ignoring")

    write_jsonl(metrics, os.path.join(args.output_dir, "per_request.jsonl"))
    summary = aggregate(metrics, wall_start=wall_start, wall_end=wall_end)
    summary["preset"] = args.preset
    summary["trace"] = args.trace
    summary["model"] = args.model
    with open(os.path.join(args.output_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    logger.info(
        "replay done: n=%d ttft_p50=%.1fms ttft_p99=%.1fms tps=%.1f cache_hit=%.3f",
        summary["n_requests"],
        summary["ttft_ms_p50"],
        summary["ttft_ms_p99"],
        summary["output_tokens_per_s"],
        summary["cache_hit_ratio"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
