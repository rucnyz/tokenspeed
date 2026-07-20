#!/usr/bin/env bash
# Wait for the in-flight capped swarm regime (old CappedFreeList-only binary)
# to finish, then deliberately SKIP the old-binary capped shifting regime
# (low marginal value: direction-bug root cause already proven at the code
# level via long_horizon's 43/43 single-direction fires; shifting is the
# slowest/most crash-prone regime and would just delay the higher-value
# fixed-binary rerun). Terminates the matrix driver early, aggregates the
# 2 completed regimes (long_horizon + swarm) for the record, then kills the
# outer run_fixed_both_matrices.sh driver so deploy_direction_fix_and_rerun.sh's
# wait loop proceeds immediately.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CAPD="$ROOT/runs/hima_matrix_fixed_capped_20260719_185041"
MATRIX_PID=2996448
DRIVER_PID=3836719
LOG="$ROOT/runs/skip_shifting.log"
log() { echo "[skip $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "waiting for swarm/sys rep3 (sys_r3.json) to land"
while [ ! -f "$CAPD/swarm/sys/sys_r3.json" ]; do sleep 15; done
log "swarm sys rep3 done; giving base a moment (already 3/3) and teardown time"
sleep 20

log "killing matrix driver pgid (prevents shifting from starting): $MATRIX_PID"
pkill -9 -P "$MATRIX_PID" 2>/dev/null
kill -9 "$MATRIX_PID" 2>/dev/null
# Also reap any stray run_arm_ts.sh / smg_grpc_servicer left from swarm teardown.
pkill -9 -f "run_arm_ts.sh sys .*swarm/sys" 2>/dev/null
pkill -9 -f "run_arm_ts.sh base .*swarm/base" 2>/dev/null
sleep 10

log "aggregating completed regimes (long_horizon + swarm) for the record"
python3 "$ROOT/scripts/aggregate_regime_matrix.py" "$CAPD" --min-valid 3 \
  --json "$CAPD/summary.json" > "$CAPD/summary.txt" 2>&1
log "summary -> $CAPD/summary.txt (shifting intentionally skipped on old binary)"
cat "$CAPD/summary.txt" | tee -a "$LOG"

log "killing outer driver so the direction-fix watcher proceeds: $DRIVER_PID"
kill -9 "$DRIVER_PID" 2>/dev/null
sleep 3
kill -0 "$DRIVER_PID" 2>/dev/null && log "WARNING: driver still alive" || log "driver exited; watcher should now proceed"
