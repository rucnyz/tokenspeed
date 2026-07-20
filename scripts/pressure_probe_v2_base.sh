#!/usr/bin/env bash
# Base-only probe on the synthetic sustained-pressure trace + a shrunk KV
# pool, to confirm real memory pressure (queue backing up, kv_free_pages
# approaching 0, cache_hit_ratio dropping) before spending time on the
# three-arm comparison. See docs/guides/hima_phase3.md for the rationale.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /data/yuzhou/projects/tokenspeed/tokenspeed/bin/activate
unset PYTHONPATH

MODEL="${MODEL:-/scratch/yuzhou/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
TRACE="${TRACE:-dataset/claude-code-traces/traces/synthetic/cc_qwen3p5_9b_sustained_pressure_v1.jsonl}"
OUT="${OUT:-runs/hima_pressure_v2_probe_20260714/base/rep0}"
TIME_SCALE="${TIME_SCALE:-0.35}"
TIMEOUT_S="${REQUEST_TIMEOUT_S:-1200}"
GPU="${GPU:-1}"
PORT="${PORT:-11500}"
# ~35% smaller than the ~2.98M-token pool the previous 3-arm run auto-sized
# to (46639 pages * block_size 64) on this GPU, so base is forced into real
# admission/eviction pressure instead of floating at <80% peak utilization.
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-1940000}"

mkdir -p "$OUT"
cmd=(
  python -u -m tokenspeed.agentreplay
  --trace "$TRACE" --model "$MODEL" --preset base
  --output-dir "$OUT"
  --time-scale "$TIME_SCALE"
  --max-inter-session-gap-s -1
  --request-timeout-s "$TIMEOUT_S"
  --base-gpu-id "$GPU" --override "port=${PORT}"
  --override "max_total_tokens=${MAX_TOTAL_TOKENS}"
)
echo "[pressure_probe_v2_base] launching: ${cmd[*]}"
echo "[pressure_probe_v2_base] output-dir=${OUT}"
exec "${cmd[@]}" > "${OUT}/run.log" 2>&1
