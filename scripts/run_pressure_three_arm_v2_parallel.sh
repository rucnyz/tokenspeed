#!/usr/bin/env bash
# Parallel three-arm (base/sys/sys_lru) comparison on the synthetic
# sustained-pressure trace + a shrunk KV pool. Runs all three arms
# concurrently on separate GPUs so total wall time ~= one arm's duration
# instead of the sum of all three, and (importantly for a fair comparison)
# all three arms see the same host-level (CPU/other-tenant) contention
# throughout, rather than base running alone and sys/sys_lru running under
# extra contention later.
#
# See scripts/build_pressure_trace.py and scripts/pressure_probe_v2_base.sh
# for how the trace/pool-size were validated to produce real, sustained
# pressure before this script existed.
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

# Distinct, currently-idle GPUs -- adjust if the host's utilization changes.
GPU_BASE="${GPU_BASE:-2}"
GPU_SYS="${GPU_SYS:-3}"
GPU_SYS_LRU="${GPU_SYS_LRU:-4}"

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
  echo "[three_arm_v2_par] $preset on GPU $gpu -> ${OUT}/${sub}/rep0"
  "${cmd[@]}" > "${OUT}/${sub}/rep0/run.log" 2>&1
  echo "[three_arm_v2_par] $preset done exit=$?"
}

run_arm base "$GPU_BASE" 11600 base &
pid_base=$!
run_arm sys "$GPU_SYS" 11610 sys "xpool_saturation_low=0.4" &
pid_sys=$!
run_arm sys_lru "$GPU_SYS_LRU" 11620 sys_lru "xpool_saturation_low=0.4" &
pid_sys_lru=$!

echo "[three_arm_v2_par] launched base(pid=$pid_base) sys(pid=$pid_sys) sys_lru(pid=$pid_sys_lru)"

wait "$pid_base" "$pid_sys" "$pid_sys_lru"

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
