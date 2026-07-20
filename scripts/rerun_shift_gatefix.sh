#!/usr/bin/env bash
# 2026-07-20: rerun ONLY the shifting regime on the GATE-FIXED scheduler binary,
# GPU3(sys)/GPU7(base), paired, N=3, uncapped -- per user request (long_horizon
# and swarm keep their drain-fix-binary results; only shifting needs the gate
# fix because it is the sole regime whose sys arm crashed).
#
# Gate fix (csrc/scheduler/scheduler.cpp newXPoolCappedDrainRetractOperations):
# proactive capped-drain retraction now only fires while a prepared fire is
# still pending its drain (kv_pre_shrunk_pages_ / mamba_pre_shrunk_slots_ > 0).
# Previously, capped tail pages/slots that merely LINGERED after a fire already
# committed (esp. static mamba_to_kv, whose capped tail slots are pinned by
# long-running decoders -> HasCappedMambaInflight stays true for minutes) kept
# driving the retract loop with no drain waiting on them: pointless and the
# source of the alloc_count=0/local_available=1 crash under conc=128 pressure.
# The .so was rebuilt + deployed into site-packages, so a fresh server boot
# picks it up.
#
# Usage: setsid bash scripts/rerun_shift_gatefix.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/tokenspeed"
export PATH="$VENV/bin:$PATH"
TRACE_DIR=${TRACE_DIR:-/data/yuzhou/projects/agentreplay/data/traces}
TRACE="$TRACE_DIR/cc_qwen_shift_composite.jsonl"
NREPS=${NREPS:-3}
CONC=${CONC:-128}
GPU_SYS=${GPU_SYS:-3}
GPU_BASE=${GPU_BASE:-7}
PORT_SYS=${PORT_SYS:-30300}
PORT_BASE=${PORT_BASE:-30700}
SYS_XPOOL_ENV=${SYS_XPOOL_ENV:-"XPOOL_DRAIN_NOPROGRESS_ABORT_S=0.5 XPOOL_DRAIN_TIMEOUT_S=2.0"}

STAMP=$(date +%Y%m%d_%H%M%S)
OUT="$ROOT/runs/hima_shift_gatefix_uncapped_${STAMP}"
echo "$OUT" > "$ROOT/runs/.last_shift_gatefix"
mkdir -p "$OUT/shifting"/{sys,base}
LOG="$OUT/rerun.log"
log() { echo "[shift-gatefix $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
log "OUT=$OUT  trace=$(basename "$TRACE")  conc=$CONC  GPU sys=$GPU_SYS base=$GPU_BASE  ports $PORT_SYS/$PORT_BASE"

wait_gpu_free() {
  local g=$1
  while :; do
    used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [ "${used:-99999}" -lt 2000 ] && return 0
    log "waiting GPU$g to free (used=${used}MiB)"; sleep 20
  done
}
wait_gpu_free "$GPU_SYS"; wait_gpu_free "$GPU_BASE"
log "GPUs free; launching paired shifting run"

# sys arm first ($SYS_XPOOL_ENV intentionally unquoted -> separate env words).
# shellcheck disable=SC2086
setsid env GPU="$GPU_SYS" PORT="$PORT_SYS" EXTRA_COMMON="" $SYS_XPOOL_ENV \
  bash "$ROOT/scripts/run_arm_ts.sh" sys "$TRACE" 0.5 "$CONC" - "$NREPS" \
  "$OUT/shifting/sys" > "$OUT/shifting/sys/runner.log" 2>&1 &
SYS_PID=$!
echo "$SYS_PID" > "$OUT/shifting/.sys_runner_pid"
sleep 20
setsid env GPU="$GPU_BASE" PORT="$PORT_BASE" EXTRA_COMMON="" \
  bash "$ROOT/scripts/run_arm_ts.sh" base "$TRACE" 0.5 "$CONC" - "$NREPS" \
  "$OUT/shifting/base" > "$OUT/shifting/base/runner.log" 2>&1 &
BASE_PID=$!
echo "$BASE_PID" > "$OUT/shifting/.base_runner_pid"
log "launched: sys_runner=$SYS_PID base_runner=$BASE_PID"

for p in "$SYS_PID" "$BASE_PID"; do
  while kill -0 "$p" 2>/dev/null; do sleep 30; done
  log "arm runner $p exited"
done

log "both arms done; aggregating shifting"
python3 "$ROOT/scripts/aggregate_regime_matrix.py" "$OUT" --min-valid "$NREPS" \
  --json "$OUT/summary.json" > "$OUT/summary.txt" 2>&1 || log "AGGREGATION FLAGGED INCOMPLETE ARMS"
log "=== summary -> $OUT/summary.txt ==="
cat "$OUT/summary.txt" | tee -a "$LOG"
