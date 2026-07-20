#!/usr/bin/env bash
# 2026-07-20: deploy the mamba_to_kv drain fix and rerun the full uncapped
# three-regime matrix.
#
# Root cause fixed (python/tokenspeed/runtime/cache/arena/xpool_actuator.py):
# a mamba_to_kv fire drained ``has_capped_mamba_inflight`` even when the mamba
# arena is *static* (logical-only, pre-mapped -- the production path).  In that
# mode no mamba memory is ever unmapped, so there is no kernel-vs-unmap race to
# wait on; worse, PrepareMambaToKvFire caps the *tail* mamba slots, which under
# smallest-id-first allocation are pinned by long-running decode sessions, so
# the capped in-flight count stalled (observed stuck at 2) and EVERY
# mamba_to_kv fire was cancelled on the no-progress abort -- silently disabling
# the KV<-mamba rebalance the budgeter direction-fix enables.  The fix skips
# the drain when the mamba arena is static.  It is a pure-Python change to the
# editable source tree, so a plain server restart (i.e. the rerun) picks it up:
# no C++ rebuild and no site-packages copy required.
#
# Flow:
#   1) wait for the in-flight (pre-drain-fix) matrix's swarm rep3 to land;
#   2) stop that matrix BEFORE it boots the shifting regime (shifting on the
#      pre-fix binary has no value -- it would replay the broken mamba drain);
#   3) aggregate its two completed regimes (long_horizon + swarm) for the record;
#   4) once GPU 3/7 are free (CPU no longer saturated), run the XPool actuator
#      test suite to VALIDATE the drain fix -- gate the rerun on it passing;
#   5) launch a fresh uncapped 3-regime (long_horizon/swarm/shifting) base-vs-sys
#      matrix on GPU3(sys)/GPU7(base) with the drain-fixed servers.
#
# Usage: setsid bash scripts/deploy_drainfix_and_rerun.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/tokenspeed"
export PATH="$VENV/bin:$PATH"

CUR="$ROOT/runs/hima_matrix_dirfix_uncapped_20260720_074308"
MATRIX_PID=${MATRIX_PID:-1562721}
LOG="$ROOT/runs/drainfix_rerun.log"
mkdir -p "$ROOT/runs"
log() { echo "[drainfix $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# 1) Wait for swarm rep3 on BOTH arms.
log "waiting for pre-fix matrix swarm rep3 (sys_r3.json + base_r3.json)"
while [ ! -f "$CUR/swarm/sys/sys_r3.json" ] || [ ! -f "$CUR/swarm/base/base_r3.json" ]; do
  sleep 20
done
log "swarm rep3 landed; letting teardown settle before stopping the matrix"
sleep 20

# 2) Stop the matrix before shifting boots.
log "killing pre-fix matrix driver pid=$MATRIX_PID (prevents shifting from starting)"
pkill -9 -P "$MATRIX_PID" 2>/dev/null
kill -9 "$MATRIX_PID" 2>/dev/null
pkill -9 -f "run_arm_ts.sh sys .*dirfix_uncapped.*swarm/sys" 2>/dev/null
pkill -9 -f "run_arm_ts.sh base .*dirfix_uncapped.*swarm/base" 2>/dev/null
# Safety: reap anything that may have started booting the shifting regime.
pkill -9 -f "dirfix_uncapped_20260720_074308/shifting" 2>/dev/null
sleep 25

# 3) Aggregate the two completed regimes for the record.
log "aggregating pre-fix matrix (long_horizon + swarm)"
python3 "$ROOT/scripts/aggregate_regime_matrix.py" "$CUR" --min-valid 3 \
  --json "$CUR/summary.json" > "$CUR/summary.txt" 2>&1 || log "aggregate flagged incomplete arms"
log "pre-fix summary -> $CUR/summary.txt"

# 4) Wait for GPU 3/7 to be free (releases CPU too), then validate the fix.
wait_gpu_free() {
  local g=$1
  while :; do
    used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [ "${used:-99999}" -lt 2000 ] && return 0
    sleep 20
  done
}
wait_gpu_free 3; wait_gpu_free 7
log "GPU 3/7 free"

log "=== validating drain fix: XPool actuator test suite ==="
PYTEST_LOG="$ROOT/runs/drainfix_pytest.log"
if ! CUDA_VISIBLE_DEVICES="" python -m pytest \
      "$ROOT/test/runtime/test_xpool_actuator_drain.py" \
      "$ROOT/test/runtime/cache/test_xpool_fire.py" \
      "$ROOT/test/runtime/test_xpool_actuator_metrics.py" \
      -q > "$PYTEST_LOG" 2>&1; then
  log "PYTEST FAILED -- see $PYTEST_LOG; NOT launching the rerun"
  tail -25 "$PYTEST_LOG" | tee -a "$LOG"
  exit 1
fi
log "pytest OK ($(grep -oE '[0-9]+ passed' "$PYTEST_LOG" | tail -1))"

# 5) Launch the fresh drain-fixed uncapped 3-regime matrix.
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="$ROOT/runs/hima_matrix_drainfix_uncapped_${STAMP}"
echo "$OUT" > "$ROOT/runs/.last_drainfix_uncapped"
log "=== launching drain-fixed uncapped 3-regime matrix -> $OUT (GPU3 sys / GPU7 base) ==="
cd "$ROOT"
CAP="" NREPS=3 GPU_SYS=3 GPU_BASE=7 bash scripts/run_regime_matrix.sh "$OUT" >> "$LOG" 2>&1
log "=== matrix complete -> $OUT/summary.txt ==="
cat "$OUT/summary.txt" 2>/dev/null | tee -a "$LOG"
