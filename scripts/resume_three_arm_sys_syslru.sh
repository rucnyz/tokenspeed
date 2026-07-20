#!/usr/bin/env bash
# Resume the pressure three-arm comparison after `base` already succeeded and
# `sys` crashed on the tree-corrupting alloc_count reconcile bug (fixed in
# scheduleRetract()/IsProactiveRetractFeasible(), 2026-07-12). Runs only
# sys and sys_lru, staggered on separate GPUs, no probe-wait needed since
# GPUs are already free.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /data/yuzhou/projects/tokenspeed/tokenspeed/bin/activate
unset PYTHONPATH

MODEL="${MODEL:-/scratch/yuzhou/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
TRACE="${TRACE:-dataset/claude-code-traces/traces/cc_qwen3p5_9b_clean.jsonl}"
OUT="runs/hima_pressure_three_arm_20260712"
TIME_SCALE="${TIME_SCALE:-0.25}"
MAX_SESSIONS="${MAX_SESSIONS:-60}"
TIMEOUT_S="${REQUEST_TIMEOUT_S:-1200}"
MIN_GPU_FREE_MIB="${MIN_GPU_FREE_MIB:-80000}"

wait_gpu_free() {
  local gpu_id="$1"
  local min_free_mib="$2"
  while true; do
    local free_mib
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | tr -d ' ')"
    if [[ -n "$free_mib" && "$free_mib" -ge "$min_free_mib" ]]; then
      echo "[resume] GPU${gpu_id} free=${free_mib}MiB (>= ${min_free_mib}MiB)"
      return 0
    fi
    echo "[resume] waiting GPU${gpu_id} free=${free_mib:-?}MiB < ${min_free_mib}MiB"
    sleep 30
  done
}

run_arm() {
  local preset="$1" gpu="$2" port="$3" sub="$4" extra="${5:-}"
  mkdir -p "${OUT}/${sub}/rep0"
  local -a cmd=(
    python -u -m tokenspeed.agentreplay
    --trace "$TRACE" --model "$MODEL" --preset "$preset"
    --output-dir "${OUT}/${sub}/rep0"
    --max-sessions "$MAX_SESSIONS" --time-scale "$TIME_SCALE"
    --request-timeout-s "$TIMEOUT_S"
    --base-gpu-id "$gpu" --override "port=${port}"
  )
  if [[ -n "$extra" ]]; then
    cmd+=(--override "$extra")
  fi
  echo "[resume] $preset on GPU $gpu"
  "${cmd[@]}" > "${OUT}/${sub}/rep0/run.log" 2>&1
  echo "[resume] $preset done exit=$?"
}

wait_gpu_free 3 "$MIN_GPU_FREE_MIB"
run_arm sys 3 11410 sys "xpool_saturation_low=0.4"
wait_gpu_free 1 "$MIN_GPU_FREE_MIB"
run_arm sys_lru 1 11420 sys_lru "xpool_saturation_low=0.4"

python3 << 'PY'
import json
from pathlib import Path
out = Path("runs/hima_pressure_three_arm_20260712")
for arm in ["base", "sys", "sys_lru"]:
    p = out / arm / "rep0" / "summary.json"
    if not p.exists():
        print(arm, "MISSING")
        continue
    s = json.load(open(p))
    print(arm, "tok/s", round(s["output_tokens_per_s"], 2),
          "ttft_p90", round(s["ttft_ms_p90"]),
          "lat_p99", round(s["latency_ms_p99"]),
          "n_failed", s.get("n_failed"))
PY
