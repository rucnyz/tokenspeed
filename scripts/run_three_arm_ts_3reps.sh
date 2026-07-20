#!/usr/bin/env bash
# Robust three-arm (base / sys / sys_lru) parallel comparison over the
# CANONICAL `ts serve` + vendored-agentreplay path, repeated REPS times, under
# genuine sustained KV scarcity so HiMA's cross-pool machinery is actually
# exercised (docs/guides/hima_phase3.md: HiMA only wins on throughput, not just
# TTFT, once KV is truly scarce).
#
# Robustness features:
#  * each arm on its own GPU + a widely-spaced port block (2000 apart). The
#    per-engine ZMQ/nccl cluster is deterministic and spans <10 ports (see the
#    PortArgs.init_new isolation fix), so 2000 spacing is belt-and-suspenders
#    isolation on top of that fix.
#  * arms launched staggered a few seconds apart, both to avoid any residual
#    port-probe TOCTOU and to spread CUDA-init memory spikes across GPUs.
#  * per-arm wall-clock watchdog (`timeout`) so a wedged arm can't hang forever.
#  * heartbeat file so you can tell "still working" from "silently dead"
#    without tailing logs.
#  * arm-level resume: an arm whose REPS reps all produced JSON is skipped, so
#    re-running the same command line after a kill resumes rather than restarts.
#  * all three arms share the SAME `--max-total-tokens` cap so the comparison is
#    fair; HiMA arms additionally get a looser saturation gate.
#
# The OpenMP/affinity workaround for this box's kernel-6.17 sched_setaffinity
# wedge is applied inside run_arm_ts.sh already.
#
# Launch detached so an SSH drop / chat gap can't reap it:
#   OUT=runs/hima_3arm_ts_$(date +%Y%m%d) MAXTOK=1200000 \
#     setsid nohup bash scripts/run_three_arm_ts_3reps.sh \
#     > runs/hima_3arm_ts_YYYYMMDD/orchestrator.log 2>&1 &
#   # if killed mid-run, re-run the same command to resume.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
source /data/yuzhou/projects/tokenspeed/tokenspeed/bin/activate
unset PYTHONPATH

TRACE="${TRACE:-/data/yuzhou/projects/agentreplay/data/traces/cc_qwen_t12.jsonl}"
CONC="${CONC:-96}"
REPS="${REPS:-3}"
STAGGER="${STAGGER:-0.5}"
LIMIT="${LIMIT:--}"
MAXTOK="${MAXTOK:?set MAXTOK to the KV-pool cap in tokens that creates scarcity}"
SAT_LOW="${SAT_LOW:-0.4}"
OUT="${OUT:-runs/hima_three_arm_ts_3reps_$(date +%Y%m%d_%H%M%S)}"
ARM_WALL_TIMEOUT_S="${ARM_WALL_TIMEOUT_S:-10800}"   # 3h watchdog per arm (boot + REPS reps)

GPU_BASE="${GPU_BASE:-0}"; GPU_SYS="${GPU_SYS:-1}"; GPU_SYS_LRU="${GPU_SYS_LRU:-2}"
PORT_BASE="${PORT_BASE:-30100}"; PORT_SYS="${PORT_SYS:-32100}"; PORT_SYS_LRU="${PORT_SYS_LRU:-34100}"

mkdir -p "$OUT"
HB="$OUT/heartbeat.txt"
beat(){ echo "$(date -Is) $*" >> "$HB"; }
echo "[3arm-ts] OUT=$OUT"
echo "[3arm-ts] TRACE=$TRACE CONC=$CONC REPS=$REPS LIMIT=$LIMIT MAXTOK=$MAXTOK SAT_LOW=$SAT_LOW"
echo "[3arm-ts] GPUs base=$GPU_BASE sys=$GPU_SYS sys_lru=$GPU_SYS_LRU ports=$PORT_BASE/$PORT_SYS/$PORT_SYS_LRU"
beat "orchestrator start OUT=$OUT MAXTOK=$MAXTOK CONC=$CONC REPS=$REPS"

arm_done(){  # $1=arm dir; return 0 iff all REPS json files exist
  local d="$1" arm r
  arm="$(basename "$d")"
  for r in $(seq 1 "$REPS"); do [ -f "$d/${arm}_r${r}.json" ] || return 1; done
  return 0
}

run_one(){
  local arm="$1" gpu="$2" port="$3" sat="$4"
  local dir="$OUT/$arm"; mkdir -p "$dir"
  if arm_done "$dir"; then
    echo "[3arm-ts] $arm already complete -- SKIP (resume)"; beat "SKIP $arm (resume)"; return 0
  fi
  beat "START $arm gpu=$gpu port=$port sat=$sat"
  EXTRA_COMMON="--max-total-tokens $MAXTOK" \
  XPOOL_SATURATION_LOW="$sat" \
  GPU="$gpu" PORT="$port" \
    timeout --kill-after=60 "$ARM_WALL_TIMEOUT_S" \
    bash scripts/run_arm_ts.sh "$arm" "$TRACE" "$STAGGER" "$CONC" "$LIMIT" "$REPS" "$dir" \
    > "$dir/runner.log" 2>&1
  local rc=$?
  beat "DONE $arm rc=$rc"
  echo "[3arm-ts] $arm finished rc=$rc"
}

# Staggered parallel launch, one GPU per arm. base's saturation value is unused
# (base has no XPool) but passed for symmetry.
run_one base    "$GPU_BASE"    "$PORT_BASE"    0.5       &  pid_base=$!
sleep 8
run_one sys     "$GPU_SYS"     "$PORT_SYS"     "$SAT_LOW" &  pid_sys=$!
sleep 8
run_one sys_lru "$GPU_SYS_LRU" "$PORT_SYS_LRU" "$SAT_LOW" &  pid_lru=$!

echo "[3arm-ts] launched base=$pid_base sys=$pid_sys sys_lru=$pid_lru"
beat "all arms launched"
wait "$pid_base" "$pid_sys" "$pid_lru"
beat "all arms finished"

echo "[3arm-ts] ===== summary ====="
for arm in base sys sys_lru; do
  echo "== $arm =="
  for r in $(seq 1 "$REPS"); do
    f="$OUT/$arm/${arm}_r${r}.json"
    if [ -f "$f" ]; then
      echo "  rep$r: $(grep -oE '\"(n_ok|n_error|throughput_tok_s|cache_hit)\": [^,}]+' "$f" | tr '\n' ' ')"
    else
      echo "  rep$r: (no json)"
    fi
  done
done
beat "COMPLETE"
echo "[3arm-ts] COMPLETE -> $OUT"
