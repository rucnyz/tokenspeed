#!/usr/bin/env bash
# 2026-07-20: parallelise the drain-fixed uncapped matrix across the 4 idle
# GPUs.  The sequential driver (run_regime_matrix.sh) runs long_horizon on
# GPU3/7 first, then swarm, then shifting -- ~3x the wall clock we need now
# that GPUs 0/1/2/6 are free.  This script keeps long_horizon running on 3/7
# (its arm runners were launched setsid, so they survive the driver being
# killed) and fans swarm + shifting out onto the idle GPUs concurrently,
# writing into the SAME output dir so all three regimes aggregate together.
#
#   regime     sys GPU  base GPU  trace                        conc  ports(sys/base)
#   ---------  -------  --------  ---------------------------  ----  ----------------
#   long_horiz    3        7      cc_qwen_t6.jsonl (running)     64   30300 / 30700
#   swarm         0        1      cc_qwen_t12.jsonl              64   31100 / 31500
#   shifting      2        6      cc_qwen_shift_composite.jsonl 128   31900 / 32300
#
# Ports are spaced 400 apart: each ts serve occupies PORT, PORT+1 (control),
# PORT+233..+240 (ZMQ/nccl cluster) and PORT+8313 (prometheus); 400 spacing
# keeps every server's footprint disjoint so run_arm_ts.sh's reap_orphans
# (which SIGKILLs whatever holds its PORT/CONTROL_PORT) never crosses arms.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$(cat "$ROOT/runs/.last_drainfix_uncapped")"
TRACE_DIR=${TRACE_DIR:-/data/yuzhou/projects/agentreplay/data/traces}
NREPS=${NREPS:-3}
MATRIX_PID=${MATRIX_PID:-3489565}   # placeholder; resolved below
SYS_XPOOL_ENV=${SYS_XPOOL_ENV:-"XPOOL_DRAIN_NOPROGRESS_ABORT_S=0.5 XPOOL_DRAIN_TIMEOUT_S=2.0"}
LOG="$ROOT/runs/parallel_remaining.log"
log() { echo "[parallel $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "OUT=$OUT"

# 1) Kill the sequential matrix driver so it never boots swarm/shifting on
#    GPU3/7 (long_horizon's setsid arm runners keep going independently).
DRV=$(pgrep -f "run_regime_matrix.sh.*$(basename "$OUT")" | head -1 || true)
if [ -n "$DRV" ]; then
  log "killing sequential matrix driver pid=$DRV (long_horizon arms survive; setsid)"
  kill -9 "$DRV" 2>/dev/null
fi
sleep 3

wait_gpu_free() {
  local g=$1
  while :; do
    used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [ "${used:-99999}" -lt 2000 ] && return 0
    log "waiting GPU$g to free (used=${used}MiB)"; sleep 20
  done
}

launch_regime() {
  local name=$1 trace=$2 conc=$3 gsys=$4 gbase=$5 psys=$6 pbase=$7
  mkdir -p "$OUT/$name"/{sys,base}
  wait_gpu_free "$gsys"; wait_gpu_free "$gbase"
  log "=== regime $name: sys GPU$gsys:$psys / base GPU$gbase:$pbase trace=$(basename "$trace") conc=$conc ==="
  # sys arm (drain/xpool tunables; $SYS_XPOOL_ENV intentionally unquoted).
  # shellcheck disable=SC2086
  setsid env GPU="$gsys" PORT="$psys" EXTRA_COMMON="" $SYS_XPOOL_ENV \
    bash "$ROOT/scripts/run_arm_ts.sh" sys "$trace" 0.5 "$conc" - "$NREPS" \
    "$OUT/$name/sys" > "$OUT/$name/sys/runner.log" 2>&1 &
  echo $! > "$OUT/$name/.sys_runner_pid"
  sleep 20
  setsid env GPU="$gbase" PORT="$pbase" EXTRA_COMMON="" \
    bash "$ROOT/scripts/run_arm_ts.sh" base "$trace" 0.5 "$conc" - "$NREPS" \
    "$OUT/$name/base" > "$OUT/$name/base/runner.log" 2>&1 &
  echo $! > "$OUT/$name/.base_runner_pid"
  log "regime $name launched: sys_runner=$(cat "$OUT/$name/.sys_runner_pid") base_runner=$(cat "$OUT/$name/.base_runner_pid")"
}

# 2) Fan swarm + shifting out onto the idle GPUs concurrently.
launch_regime swarm    "$TRACE_DIR/cc_qwen_t12.jsonl"            64  0 1 31100 31500
launch_regime shifting "$TRACE_DIR/cc_qwen_shift_composite.jsonl" 128 2 6 31900 32300

# 3) Wait for ALL arm runners (long_horizon on 3/7 + swarm + shifting).
log "waiting for all regime arm runners to finish"
PIDS=()
# long_horizon runners are already running (setsid children of the dead driver)
_ob="$(basename "$OUT")"
for p in $(pgrep -f "run_arm_ts.sh sys .*${_ob}/long_horizon/sys" || true) \
         $(pgrep -f "run_arm_ts.sh base .*${_ob}/long_horizon/base" || true); do PIDS+=("$p"); done
for reg in swarm shifting; do
  PIDS+=("$(cat "$OUT/$reg/.sys_runner_pid")" "$(cat "$OUT/$reg/.base_runner_pid")")
done
log "tracking arm runner pids: ${PIDS[*]}"
for p in "${PIDS[@]}"; do
  while kill -0 "$p" 2>/dev/null; do sleep 30; done
  log "arm runner $p exited"
done

# 4) Aggregate all three regimes.
log "all arms done; aggregating $OUT"
python3 "$ROOT/scripts/aggregate_regime_matrix.py" "$OUT" --min-valid "$NREPS" \
  --json "$OUT/summary.json" > "$OUT/summary.txt" 2>&1 || log "AGGREGATION FLAGGED INCOMPLETE ARMS"
log "=== summary -> $OUT/summary.txt ==="
cat "$OUT/summary.txt" | tee -a "$LOG"
