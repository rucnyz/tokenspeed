#!/usr/bin/env bash
# Monitor the three-arm x N-reps sweep launched by run_three_arm_3reps.sh.
#
# For every arm/rep it reports: run state (running/done/CRASH/pending), the
# latest and minimum kv_free_pages, how many ticks dipped into the danger zone
# (<50, where the old KV-arena bug used to fault), tick count, and any crash
# signature. Pass a number to poll every N seconds (watch mode); omit for a
# single snapshot.
#
# Usage:
#   scripts/monitor_three_arm.sh            # one snapshot
#   scripts/monitor_three_arm.sh 30         # refresh every 30s
#   OUT=runs/... scripts/monitor_three_arm.sh 30
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT:-$ROOT/runs/hima_three_arm_3reps_20260716}"
REPS="${REPS:-3}"
INTERVAL="${1:-0}"

snapshot() {
  echo "===================== $(date -Is) ====================="
  echo "OUT=$OUT"
  # GPU line
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null \
    | awk -F', ' '{printf "  gpu%s %s %s util\n",$1,$2,$3}' | paste -sd' ' -
  printf "%-8s %-5s %-9s %8s %8s %7s %7s %s\n" \
    ARM REP STATE kv_now kv_min "<50" ticks note
  for arm in base sys sys_lru; do
    for rep in $(seq 0 $((REPS-1))); do
      d="$OUT/$arm/rep$rep"
      log="$d/run.log"; bt="$d/budget_tick.log"; sm="$d/summary.json"
      [[ ! -e "$log" && ! -e "$d" ]] && { printf "%-8s %-5s %-9s\n" "$arm" "$rep" "pending"; continue; }
      state="running"; note=""
      if grep -qiE "illegal memory|CUDA error|Traceback \(most recent" "$log" 2>/dev/null; then
        state="CRASH"; note=$(grep -iE "illegal memory|RuntimeError" "$log" 2>/dev/null | tail -1 | cut -c1-48)
      elif [[ -e "$sm" ]]; then
        state="done"
      elif ! pgrep -f "$d" >/dev/null 2>&1 && [[ -s "$log" ]]; then
        # no live process and not done -> likely exited without summary
        if grep -qiE "port=.*already|Address already" "$log" 2>/dev/null; then
          state="port-busy"
        fi
      fi
      kv_now="-"; kv_min="-"; lt50="-"; ticks="0"
      if [[ -s "$bt" ]]; then
        mapfile -t kvs < <(grep -oE 'kv_free=[0-9]+' "$bt" 2>/dev/null | sed 's/kv_free=//')
        ticks=${#kvs[@]}
        if (( ticks > 0 )); then
          kv_now=${kvs[-1]}
          kv_min=$(printf '%s\n' "${kvs[@]}" | sort -n | head -1)
          lt50=$(printf '%s\n' "${kvs[@]}" | awk '$1<50' | wc -l | tr -d ' ')
        fi
      fi
      printf "%-8s %-5s %-9s %8s %8s %7s %7s %s\n" \
        "$arm" "$rep" "$state" "$kv_now" "$kv_min" "$lt50" "$ticks" "$note"
    done
  done
}

if (( INTERVAL > 0 )); then
  while :; do
    clear 2>/dev/null || true
    snapshot
    sleep "$INTERVAL"
  done
else
  snapshot
fi
