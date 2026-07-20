#!/usr/bin/env bash
# Three-arm (base/sys/sys_lru) comparison on the synthetic sustained-pressure
# trace + a shrunk KV pool (see scripts/build_pressure_trace.py and
# scripts/pressure_probe_v2_base.sh). Only launch this after the base-only
# probe has confirmed real, sustained pressure (queue_len backing up,
# kv_free_pages getting genuinely low for a large fraction of the run) --
# see docs/guides/hima_phase3.md.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /data/yuzhou/projects/tokenspeed/tokenspeed/bin/activate
unset PYTHONPATH

MODEL="${MODEL:-/scratch/yuzhou/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
TRACE="${TRACE:-dataset/claude-code-traces/traces/synthetic/cc_qwen3p5_9b_sustained_pressure_v1.jsonl}"
OUT="${OUT:-runs/hima_pressure_three_arm_v2_20260714}"
TIME_SCALE="${TIME_SCALE:-0.35}"
TIMEOUT_S="${REQUEST_TIMEOUT_S:-1200}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-1940000}"
MIN_GPU_FREE_MIB="${MIN_GPU_FREE_MIB:-80000}"

wait_gpu_free() {
  local gpu_id="$1"
  local min_free_mib="$2"
  while true; do
    local free_mib
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | tr -d ' ')"
    if [[ -n "$free_mib" && "$free_mib" -ge "$min_free_mib" ]]; then
      echo "[three_arm_v2] GPU${gpu_id} free=${free_mib}MiB (>= ${min_free_mib}MiB)"
      return 0
    fi
    echo "[three_arm_v2] waiting GPU${gpu_id} free=${free_mib:-?}MiB < ${min_free_mib}MiB"
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
    --time-scale "$TIME_SCALE"
    --max-inter-session-gap-s -1
    --request-timeout-s "$TIMEOUT_S"
    --base-gpu-id "$gpu" --override "port=${port}"
    --override "max_total_tokens=${MAX_TOTAL_TOKENS}"
  )
  if [[ -n "$extra" ]]; then
    cmd+=(--override "$extra")
  fi
  echo "[three_arm_v2] $preset on GPU $gpu"
  "${cmd[@]}" > "${OUT}/${sub}/rep0/run.log" 2>&1
  echo "[three_arm_v2] $preset done exit=$?"
}

wait_gpu_free 1 "$MIN_GPU_FREE_MIB"
run_arm base 1 11600 base
wait_gpu_free 3 "$MIN_GPU_FREE_MIB"
run_arm sys 3 11610 sys "xpool_saturation_low=0.4"
wait_gpu_free 1 "$MIN_GPU_FREE_MIB"
run_arm sys_lru 1 11620 sys_lru "xpool_saturation_low=0.4"

python3 << PY
import json
from pathlib import Path
out = Path("${OUT}")
for arm in ["base", "sys", "sys_lru"]:
    p = out / arm / "rep0" / "summary.json"
    if not p.exists():
        print(arm, "MISSING")
        continue
    s = json.load(open(p))
    print(arm, "tok/s", round(s["output_tokens_per_s"], 2),
          "ttft_p90", round(s["ttft_ms_p90"]),
          "lat_p99", round(s["latency_ms_p99"]),
          "cache_hit", round(s["cache_hit_ratio"], 3))
PY
