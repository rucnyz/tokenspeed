#!/usr/bin/env bash
# Three-arm (base/sys/sys_lru) sustained-pressure comparison, run in parallel
# on three GPUs and repeated REPS times for statistical significance.
#
# Designed to survive being killed mid-run (this box has been observed to
# reap long-lived background process groups after multi-hour idle gaps
# between chat turns -- see docs/guides/hima_phase3.md "Environment
# resilience"). Re-invoking this exact script after a kill RESUMES rather
# than restarts: any arm/rep that already has a summary.json is skipped, and
# a heartbeat file lets you tell "still working" apart from "silently dead"
# without reading logs.
#
# Each rep launches all three arms concurrently (one GPU each) so every arm
# sees identical host-level contention, then waits for all three before the
# next rep. Between reps we wait for the GPUs to actually free their memory
# so a lingering scheduler child never OOMs the next rep.
#
# Includes the OpenMP/affinity workaround for the kernel sched_setaffinity
# wedge observed on this box (kernel 6.17, load spikes): without disabling
# OpenMP thread binding the engine can hang unkillable in affine_move_task
# during CUDA init before it prints anything.
#
# The KV-arena sub-chunk shrink fix (kv_arena.py) is picked up automatically
# via the editable install.
#
# Usage:
#   scripts/run_three_arm_3reps.sh                       # 3 reps on GPU 2/3/4
#   REPS=3 GPU_BASE=2 GPU_SYS=3 GPU_SYS_LRU=4 TRACE=... scripts/run_three_arm_3reps.sh
#   # if killed mid-run, just re-run the same command line to resume.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /data/yuzhou/projects/tokenspeed/tokenspeed/bin/activate
unset PYTHONPATH

# --- OpenMP / affinity workaround (see header) --------------------------------
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=false
export KMP_AFFINITY=disabled
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=false
export NUMEXPR_NUM_THREADS=1

MODEL="${MODEL:-/scratch/yuzhou/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
TRACE="${TRACE:-dataset/claude-code-traces/traces/synthetic/cc_qwen3p5_9b_sustained_pressure_v3_small.jsonl}"
OUT="${OUT:-runs/hima_three_arm_3reps_v3small_20260717}"
TIME_SCALE="${TIME_SCALE:-0.35}"
MAX_SESSIONS="${MAX_SESSIONS:-}"   # empty -> use every session in TRACE
TIMEOUT_S="${REQUEST_TIMEOUT_S:-1200}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-1940000}"
REPS="${REPS:-3}"
ARM_WALL_TIMEOUT_S="${ARM_WALL_TIMEOUT_S:-5400}"   # 90 min watchdog per arm

GPU_BASE="${GPU_BASE:-2}"
GPU_SYS="${GPU_SYS:-3}"
GPU_SYS_LRU="${GPU_SYS_LRU:-4}"

mkdir -p "$OUT"
HEARTBEAT="$OUT/heartbeat.txt"
echo "[3reps] OUT=$OUT REPS=$REPS TRACE=$TRACE GPUs base=$GPU_BASE sys=$GPU_SYS sys_lru=$GPU_SYS_LRU"

beat() { echo "$(date -Is) $*" >> "$HEARTBEAT"; }

wait_gpu_free() {
  local gpu="$1" thresh="${2:-2000}" tries=0
  while :; do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ')
    [[ -z "$used" ]] && used=0
    if (( used < thresh )); then return 0; fi
    tries=$((tries+1))
    if (( tries > 120 )); then
      echo "[3reps] WARN gpu $gpu still $used MiB after 10min; proceeding anyway"
      return 0
    fi
    sleep 5
  done
}

run_arm() {
  local preset="$1" gpu="$2" port="$3" sub="$4" rep="$5" extra="${6:-}"
  local dir="${OUT}/${sub}/rep${rep}"
  mkdir -p "$dir"
  if [[ -f "$dir/summary.json" ]]; then
    echo "[3reps] rep${rep} $preset already has summary.json -- SKIP (resume)"
    beat "SKIP $preset rep$rep (already done)"
    return 0
  fi
  local -a cmd=(
    python -u -m tokenspeed.agentreplay
    --trace "$TRACE" --model "$MODEL" --preset "$preset"
    --output-dir "$dir"
    --time-scale "$TIME_SCALE"
    --max-inter-session-gap-s -1
    --request-timeout-s "$TIMEOUT_S"
    --base-gpu-id "$gpu" --override "port=${port}"
    --override "max_total_tokens=${MAX_TOTAL_TOKENS}"
  )
  [[ -n "$MAX_SESSIONS" ]] && cmd+=(--max-sessions "$MAX_SESSIONS")
  [[ -n "$extra" ]] && cmd+=(--override "$extra")
  echo "[3reps] rep${rep} $preset on GPU $gpu -> $dir"
  beat "START $preset rep$rep gpu=$gpu"
  TOKENSPEED_BUDGET_LOG="${dir}/budget_tick.log" \
    timeout --kill-after=60 "$ARM_WALL_TIMEOUT_S" "${cmd[@]}" > "${dir}/run.log" 2>&1
  local rc=$?
  if [[ -f "$dir/summary.json" ]]; then
    beat "DONE $preset rep$rep exit=$rc (summary written)"
  elif (( rc == 124 || rc == 137 )); then
    beat "TIMEOUT/KILLED $preset rep$rep exit=$rc after ${ARM_WALL_TIMEOUT_S}s (no summary; will retry on re-run)"
  else
    beat "EXITED $preset rep$rep exit=$rc (no summary -- check run.log)"
  fi
  echo "[3reps] rep${rep} $preset done exit=$rc"
}

for rep in $(seq 0 $((REPS-1))); do
  base_done="$OUT/base/rep${rep}/summary.json"
  sys_done="$OUT/sys/rep${rep}/summary.json"
  lru_done="$OUT/sys_lru/rep${rep}/summary.json"
  if [[ -f "$base_done" && -f "$sys_done" && -f "$lru_done" ]]; then
    echo "[3reps] rep${rep} already fully done -- SKIP"
    continue
  fi

  echo "[3reps] ===== rep${rep} start $(date -Is) ====="
  beat "===== rep${rep} start ====="
  wait_gpu_free "$GPU_BASE"; wait_gpu_free "$GPU_SYS"; wait_gpu_free "$GPU_SYS_LRU"

  base_port=$((11600 + rep*10))
  sys_port=$((11601 + rep*10))
  lru_port=$((11602 + rep*10))

  run_arm base    "$GPU_BASE"    "$base_port" base    "$rep" &
  pid_base=$!
  run_arm sys     "$GPU_SYS"     "$sys_port"  sys     "$rep" "xpool_saturation_low=0.4" &
  pid_sys=$!
  run_arm sys_lru "$GPU_SYS_LRU" "$lru_port"  sys_lru "$rep" "xpool_saturation_low=0.4" &
  pid_lru=$!

  echo "[3reps] rep${rep} launched base=$pid_base sys=$pid_sys sys_lru=$pid_lru"
  wait "$pid_base" "$pid_sys" "$pid_lru"
  echo "[3reps] ===== rep${rep} done $(date -Is) ====="
  beat "===== rep${rep} done ====="
done

echo "[3reps] all reps attempted; aggregating"
beat "aggregating"
python3 scripts/aggregate_three_arm.py "$OUT" "$REPS" || true
echo "[3reps] COMPLETE"
beat "COMPLETE"
