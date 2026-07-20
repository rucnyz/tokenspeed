#!/usr/bin/env bash
# Wait for Phase-1d base to finish, then run pressure probe on freed GPU1.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /data/yuzhou/projects/tokenspeed/tokenspeed/bin/activate
unset PYTHONPATH

MODEL="${MODEL:-/scratch/yuzhou/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
TRACE="${TRACE:-dataset/claude-code-traces/traces/cc_qwen3p5_9b_clean.jsonl}"
OUT_PROBE="runs/hima_pressure_probe_20260712"

while pgrep -f "output-dir runs/hima_p1d_validate_20260712/base/rep0" >/dev/null 2>&1; do
  sleep 60
done
sleep 15

mkdir -p "${OUT_PROBE}/base/rep0"
echo "[probe] starting base pressure run on GPU1"
python -u -m tokenspeed.agentreplay \
  --trace "$TRACE" --model "$MODEL" --preset base \
  --output-dir "${OUT_PROBE}/base/rep0" \
  --max-sessions 60 --time-scale 0.25 --request-timeout-s 1200 \
  --base-gpu-id 1 --override port=11300 \
  > "${OUT_PROBE}/base/rep0/run.log" 2>&1
echo "[probe] done exit=$?"

python3 << 'PY'
import json, statistics
path = "runs/hima_pressure_probe_20260712/base/rep0/budget.jsonl"
rows = []
try:
    with open(path) as f:
        for line in f:
            try: rows.append(json.loads(line))
            except: pass
except FileNotFoundError:
    print("no budget.jsonl")
    raise SystemExit(1)
kv_free = [r.get("kv_free_pages", 0) for r in rows]
queue = [r.get("queue_len", 0) for r in rows]
print("kv_free min/mean", min(kv_free), statistics.mean(kv_free))
print("queue_len max/mean", max(queue), statistics.mean(queue))
PY
