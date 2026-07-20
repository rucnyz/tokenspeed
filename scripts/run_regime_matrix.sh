#!/usr/bin/env bash
# Three-regime HiMA evaluation matrix (paper table): base vs sys, N=3 paired
# reps per regime, token-exact Claude Code trace replays via run_arm_ts.sh.
#
#   regime        trace                       conc  kv cap    rationale
#   ------------  --------------------------  ----  --------  ------------------
#   long_horizon  cc_qwen_t6.jsonl             64   640000    few super-long sessions overflow KV
#   swarm         cc_qwen_t12.jsonl            64   640000    1789 sessions on same cap -> high eviction churn
#   shifting      cc_qwen_shift_composite      128  640000    workload shape flips mid-run
#
# Pairing: for each regime, base (GPU $GPU_BASE) and sys (GPU $GPU_SYS) run
# simultaneously against the same trace/conc/cap -- same wall-clock window,
# same host, so the N=3 reps are paired samples.
#
# long_horizon may already exist (the drain-fix validation run doubles as it);
# pass SKIP_LONG_HORIZON=1 to reuse it.
#
# Usage: setsid bash scripts/run_regime_matrix.sh <outdir_root> [wait_pid ...]
#   wait_pid: optional runner PIDs to wait on before starting (lets the matrix
#   queue behind an in-flight validation run without clobbering its GPUs).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT=${1:?outdir_root}; shift || true
GPU_SYS=${GPU_SYS:-3}
GPU_BASE=${GPU_BASE:-7}
# Ports are overridable so a second matrix can run concurrently on different
# GPUs without run_arm_ts.sh's reap_orphans() killing the other matrix's
# server (it SIGKILLs whatever already holds $PORT/$CONTROL_PORT on boot).
PORT_SYS=${PORT_SYS:-30300}
PORT_BASE=${PORT_BASE:-30700}
# CAP: artificial --max-total-tokens ceiling to induce sustained KV scarcity.
# Default 640000 (the scarcity-hunting config). Pass CAP="" (explicitly empty,
# note: "-" not ":-" so an explicit empty string is NOT overridden) to omit
# the flag entirely and let the KV/mamba pool size itself from
# --mem-fraction-static like sglang/reproduce/RQ1/run_arm.sh's canonical
# (uncapped) case1/case2 config -- see docs/guides/hima_phase3.md.
CAP=${CAP-640000}
NREPS=${NREPS:-3}
TRACE_DIR=${TRACE_DIR:-/data/yuzhou/projects/agentreplay/data/traces}
SKIP_LONG_HORIZON=${SKIP_LONG_HORIZON:-0}

# XPool actuator drain/cooldown tunables applied to the SYS arm only (base has
# no actuator).  These are read by EngineCore._init_xpool_actuator via env and
# forwarded to XPoolActuator, so they tune the fire drain policy without a
# recompile.  Default (2026-07-19): bound the capacity a doomed fire holds --
# the no-progress abort releases the prepare tail-cap ~0.5 s after the capped
# in-flight count stops decreasing (was: held for the full 5 s timeout), which
# is the fix for sys's throughput/tail-TTFT regression vs base (40-66% of
# fires never drain).  Override to sweep, e.g.
#   SYS_XPOOL_ENV="XPOOL_DRAIN_NOPROGRESS_ABORT_S=0.25 XPOOL_DRAIN_TIMEOUT_S=1.0"
SYS_XPOOL_ENV=${SYS_XPOOL_ENV:-"XPOOL_DRAIN_NOPROGRESS_ABORT_S=0.5 XPOOL_DRAIN_TIMEOUT_S=2.0"}

mkdir -p "$OUT"
log() { echo "[matrix $(date +%H:%M:%S)] $*" | tee -a "$OUT/matrix.log"; }

# Queue behind any in-flight runner PIDs handed to us.
for pid in "$@"; do
  while kill -0 "$pid" 2>/dev/null; do sleep 30; done
  log "wait_pid $pid exited"
done

# Belt-and-braces: also wait until both GPUs are actually free.
wait_gpu_free() {
  local g=$1
  while :; do
    used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    [ "${used:-99999}" -lt 2000 ] && return 0
    sleep 30
  done
}
wait_gpu_free "$GPU_SYS"; wait_gpu_free "$GPU_BASE"
log "GPUs $GPU_SYS/$GPU_BASE free; starting matrix"
log "sys xpool tunables: ${SYS_XPOOL_ENV:-<none>}"

run_regime() {
  local name=$1 trace=$2 conc=$3
  local cap_extra=""
  [ -n "$CAP" ] && cap_extra="--max-total-tokens $CAP"
  log "=== regime $name: trace=$trace conc=$conc cap=${CAP:-<uncapped>} nreps=$NREPS ==="
  mkdir -p "$OUT/$name"/{sys,base}
  # Paired launch: sys first (JIT warm from prior arm), base 20 s later.
  # $SYS_XPOOL_ENV is intentionally unquoted so its "NAME=VALUE NAME2=VALUE2"
  # words become separate env assignments (sys arm only).
  # shellcheck disable=SC2086
  setsid env GPU="$GPU_SYS" PORT="$PORT_SYS" EXTRA_COMMON="$cap_extra" \
    $SYS_XPOOL_ENV \
    bash "$ROOT/scripts/run_arm_ts.sh" sys "$trace" 0.5 "$conc" - "$NREPS" \
    "$OUT/$name/sys" > "$OUT/$name/sys/runner.log" 2>&1 &
  local sys_pid=$!
  sleep 20
  setsid env GPU="$GPU_BASE" PORT="$PORT_BASE" EXTRA_COMMON="$cap_extra" \
    bash "$ROOT/scripts/run_arm_ts.sh" base "$trace" 0.5 "$conc" - "$NREPS" \
    "$OUT/$name/base" > "$OUT/$name/base/runner.log" 2>&1 &
  local base_pid=$!
  log "regime $name: sys_pid=$sys_pid base_pid=$base_pid"
  wait "$sys_pid" "$base_pid" 2>/dev/null
  log "regime $name done"
  # Give the teardown a moment to release GPU memory before the next regime.
  sleep 30; wait_gpu_free "$GPU_SYS"; wait_gpu_free "$GPU_BASE"
}

if [ "$SKIP_LONG_HORIZON" != "1" ]; then
  run_regime long_horizon "$TRACE_DIR/cc_qwen_t6.jsonl" 64
else
  log "long_horizon skipped (reusing validation run)"
fi
run_regime swarm    "$TRACE_DIR/cc_qwen_t12.jsonl" 64
run_regime shifting "$TRACE_DIR/cc_qwen_shift_composite.jsonl" 128

log "matrix complete; aggregating"
python3 "$ROOT/scripts/aggregate_regime_matrix.py" "$OUT" --min-valid "$NREPS" \
  --json "$OUT/summary.json" > "$OUT/summary.txt" 2>&1 || log "AGGREGATION FLAGGED INCOMPLETE ARMS"
log "summary -> $OUT/summary.txt"
