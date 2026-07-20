#!/usr/bin/env bash
# One-shot driver (2026-07-19): run the three-regime base-vs-sys matrix on the
# FIXED scheduler binary (CappedFreeList O(log n) rewrite) twice --
#   1) uncapped (reuses the long_horizon fix-validation, runs swarm+shifting)
#   2) capped=640000 (all three regimes fresh)
# -- sequentially on the same GPU pair. Ports are non-default so run_arm_ts.sh's
# reap_orphans() never SIGKILLs any *other* matrix still running on 30300/30700.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UNCAP="$(cat "$ROOT/runs/.last_fixed_uncapped")"
CAPD="$(cat "$ROOT/runs/.last_fixed_capped")"

export GPU_SYS=${GPU_SYS:-6}
export GPU_BASE=${GPU_BASE:-2}
export PORT_SYS=${PORT_SYS:-31800}
export PORT_BASE=${PORT_BASE:-31900}
export NREPS=${NREPS:-3}

echo "[both $(date +%H:%M:%S)] === UNCAPPED matrix (reuse long_horizon) -> $UNCAP ==="
CAP="" SKIP_LONG_HORIZON=1 bash "$ROOT/scripts/run_regime_matrix.sh" "$UNCAP"

echo "[both $(date +%H:%M:%S)] === CAPPED=640000 matrix (all fresh) -> $CAPD ==="
CAP=640000 SKIP_LONG_HORIZON=0 bash "$ROOT/scripts/run_regime_matrix.sh" "$CAPD"

echo "[both $(date +%H:%M:%S)] === BOTH MATRICES DONE ==="
echo "uncapped summary: $UNCAP/summary.txt"
echo "capped   summary: $CAPD/summary.txt"
