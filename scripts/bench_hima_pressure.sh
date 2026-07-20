#!/usr/bin/env bash
# HiMA pressure workload: higher concurrency via time_scale + max_sessions.
# Usage:
#   ./scripts/bench_hima_pressure.sh probe          # base only, check KV pressure
#   ./scripts/bench_hima_pressure.sh light_validate # Phase 1d: max_sessions=12
#   ./scripts/bench_hima_pressure.sh three_arm      # base/sys/sys_lru under pressure
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /data/yuzhou/projects/tokenspeed/tokenspeed/bin/activate
unset PYTHONPATH

MODEL="${MODEL:-/scratch/yuzhou/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
TRACE="${TRACE:-dataset/claude-code-traces/traces/cc_qwen3p5_9b_clean.jsonl}"
RUN_TAG="${RUN_TAG:-hima_pressure_$(date +%Y%m%d)}"
OUT="runs/${RUN_TAG}"

TIME_SCALE="${TIME_SCALE:-0.25}"
MAX_SESSIONS="${MAX_SESSIONS:-60}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-1200}"
BASE_GPU_BASE="${BASE_GPU_BASE:-1}"

run_one() {
  local preset="$1"
  local gpu="$2"
  local port="$3"
  local subdir="$4"
  local extra_override="${5:-}"
  mkdir -p "${OUT}/${subdir}/rep0"
  local cmd=(
    python -u -m tokenspeed.agentreplay
    --trace "$TRACE"
    --model "$MODEL"
    --preset "$preset"
    --output-dir "${OUT}/${subdir}/rep0"
    --max-sessions "$MAX_SESSIONS"
    --time-scale "$TIME_SCALE"
    --request-timeout-s "$REQUEST_TIMEOUT_S"
    --base-gpu-id "$gpu"
    --override "port=${port}"
  )
  if [[ -n "$extra_override" ]]; then
    cmd+=(--override "$extra_override")
  fi
  echo "[bench] ${preset} -> GPU ${gpu} port ${port} dir ${OUT}/${subdir}/rep0"
  "${cmd[@]}" > "${OUT}/${subdir}/rep0/run.log" 2>&1
  echo "[bench] ${preset} done exit=$?"
}

mode="${1:-probe}"
case "$mode" in
  probe)
    run_one base "$BASE_GPU_BASE" 11000 base
    ;;
  light_validate)
    MAX_SESSIONS=12 TIME_SCALE=1.0 OUT="runs/${RUN_TAG}_p1d" run_one base "$BASE_GPU_BASE" 11010 base
    run_one sys "$((BASE_GPU_BASE + 2))" 11020 sys
    ;;
  three_arm)
    run_one base "$BASE_GPU_BASE" 11100 base
    run_one sys "$((BASE_GPU_BASE + 2))" 11110 sys \
      "xpool_saturation_low=0.4"
    run_one sys_lru "$((BASE_GPU_BASE + 4))" 11120 sys_lru \
      "xpool_saturation_low=0.4"
    ;;
  *)
    echo "Unknown mode: $mode (probe|light_validate|three_arm)" >&2
    exit 1
    ;;
esac
