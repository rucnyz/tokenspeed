#!/usr/bin/env bash
# One-shot crash diagnosis run for the residual sys IMA under KV scarcity.
#
# Runs a single sys rep of the long_horizon workload with:
#   * CUDA_LAUNCH_BLOCKING=1  -- makes the CUDA driver report the illegal
#     memory access at the actual faulting kernel launch instead of at the
#     next copy_event.synchronize(), so the traceback names the real kernel.
#   * DEBUG_MEM=1             -- turns on Scheduler::check_device_mem() every
#     tick (double-owned tail page detection) plus allocator debug logs.
#
# Perf numbers from this run are meaningless (launch blocking serializes the
# GPU); the only goal is the crash signature. If the run survives the full
# rep, that is also signal: the race needs async launches to manifest.
#
# Usage: setsid bash scripts/run_crash_diag.sh <outdir> [gpu] [trace]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT=${1:?outdir}
GPU=${2:-3}
TRACE=${3:-/data/yuzhou/projects/agentreplay/data/traces/cc_qwen_t6.jsonl}
mkdir -p "$OUT"

CUDA_LAUNCH_BLOCKING=1 DEBUG_MEM=1 \
  GPU="$GPU" PORT=30500 EXTRA_COMMON="--max-total-tokens 640000" \
  NREPS=1 RETRY_BUDGET=0 \
  bash "$ROOT/scripts/run_arm_ts.sh" sys "$TRACE" 0.5 64 - 1 "$OUT" \
  > "$OUT/runner.log" 2>&1

echo "diag done -> $OUT"
grep -n "illegal memory access\|DEVICE TAIL PAGE OVERLAP\|Scheduler hit an exception" \
  "$OUT"/server_sys.log | head -40 || true
