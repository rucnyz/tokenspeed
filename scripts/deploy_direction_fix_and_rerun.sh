#!/usr/bin/env bash
# Wait for the currently in-flight capped/uncapped base-vs-sys matrix
# (run_fixed_both_matrices.sh, driver PID passed as $1) to finish, then:
#   1) rebuild + run the C++ suite to validate the mamba_to_kv direction-bug
#      fix in budget_agent.cpp (source already fixed + regression test added,
#      but not yet compiled into the .so the running servers loaded);
#   2) rebuild the python extension and deploy it into the venv site-packages
#      (editable install loads from site-packages, NOT build-py -- copying is
#      required, see docs/guides/hima_phase3.md);
#   3) launch a fresh capped=640000 three-regime (long_horizon/swarm/shifting)
#      base-vs-sys matrix on GPU3(sys)/GPU7(base) -- run_regime_matrix.sh
#      defaults -- to verify the fix lets sys win KV-bound regimes.
#
# Usage: setsid bash scripts/deploy_direction_fix_and_rerun.sh <wait_pid> [out_root]
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WAIT_PID=${1:?wait_pid}
OUT_ROOT=${2:-"$ROOT/runs/hima_matrix_direction_fix"}
VENV="$ROOT/tokenspeed"
export PATH="$VENV/bin:$PATH"

LOG="$ROOT/runs/deploy_direction_fix.log"
mkdir -p "$ROOT/runs"
log() { echo "[deploy $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "waiting for driver pid $WAIT_PID (in-flight uncapped+capped matrix) to exit"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
log "pid $WAIT_PID exited; sleeping 30s for GPU teardown"
sleep 30

cd "$ROOT/tokenspeed-scheduler"

log "=== rebuilding C++ test suite (incremental) ==="
if ! cmake --build build-tests --target tokenspeed_scheduler_tests -j 8 2>&1 | tee -a "$LOG"; then
  log "C++ BUILD FAILED -- aborting deploy, matrix NOT started"
  exit 1
fi

log "=== running full C++ test suite ==="
./build-tests/tokenspeed_scheduler_tests 2>&1 | tee -a "$LOG"
CPPRC=${PIPESTATUS[0]:-1}
if [ "$CPPRC" != "0" ]; then
  log "C++ TESTS FAILED (rc=$CPPRC) -- aborting deploy, matrix NOT started"
  exit 1
fi
log "C++ tests OK"

log "=== rebuilding python extension (incremental) ==="
if ! cmake --build build-py --target tokenspeed_scheduler_ext -j 8 2>&1 | tee -a "$LOG"; then
  log "PYTHON EXT BUILD FAILED -- aborting deploy, matrix NOT started"
  exit 1
fi

SP="$VENV/lib/python3.12/site-packages/tokenspeed_scheduler"
SRC="build-py/tokenspeed_scheduler_ext.cpython-312-x86_64-linux-gnu.so"
if [ ! -f "$SRC" ]; then
  log "BUILT .so NOT FOUND at $SRC -- aborting deploy"
  exit 1
fi
log "deploying $SRC -> $SP"
cp "$SRC" "$SP/tokenspeed_scheduler_ext.cpython-312-x86_64-linux-gnu.so"

log "=== verifying import of freshly deployed extension ==="
if ! "$VENV/bin/python" -c "from tokenspeed_scheduler import Scheduler; print('import OK')" 2>&1 | tee -a "$LOG"; then
  log "IMPORT FAILED -- aborting matrix launch"
  exit 1
fi

STAMP=$(date +%Y%m%d_%H%M%S)
OUT="${OUT_ROOT}_${STAMP}"
echo "$OUT" > "$ROOT/runs/.last_direction_fix_capped"
log "=== launching fixed-binary capped=640000 3-regime matrix -> $OUT (GPU_SYS=3 sys / GPU_BASE=7 base) ==="
cd "$ROOT"
CAP=640000 NREPS=3 bash scripts/run_regime_matrix.sh "$OUT" 2>&1 | tee -a "$LOG"
log "=== matrix complete -> $OUT/summary.txt ==="
