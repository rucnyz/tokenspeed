#!/usr/bin/env bash
# Periodically summarize a budget.jsonl file's key pressure signals.
# Usage: monitor_budget.sh <budget.jsonl> [interval_s]
set -euo pipefail
FILE="$1"
INTERVAL="${2:-60}"
while true; do
  if [[ -f "$FILE" ]]; then
    python3 -c "
import json
try:
    with open('$FILE') as f:
        lines = f.readlines()
except FileNotFoundError:
    raise SystemExit
if not lines:
    raise SystemExit
r = json.loads(lines[-1])
print(f\"t={r['t']:.0f}s decoding={r['decoding']} prefilling={r['prefilling']} \"
      f\"queue={r['queue_len']} kv_free={r['kv_free_pages']} kv_active={r['kv_active_pages']} \"
      f\"retract={r['retract_count']}\")
"
  else
    echo "waiting for $FILE to appear..."
  fi
  sleep "$INTERVAL"
done
