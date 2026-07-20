#!/usr/bin/env bash
# Boot one tokenspeed ARM via `ts serve`, run N vendored-agentreplay reps
# against its /generate endpoint (control HTTP server), tear down.
#
# Aligns with sglang/reproduce/RQ1/run_arm.sh methodology:
# - same Claude-Code closed-loop harness (vendored under
#   third_party/agentreplay — a COPY of /data/yuzhou/projects/agentreplay,
#   so the shared sglang package is never modified)
# - same canonical corpus traces (cc_qwen_t6 / cc_qwen_t12)
# - pressure from --max-concurrency (asyncio.Semaphore), NOT synthetic
#   session duplication
#
# Usage:
#   scripts/run_arm_ts.sh <base|sys|sys_lru> <trace> <stagger|-> <maxconc> <limit|-> <nreps> <outdir>
#
# stagger: pass 0.5 to match sglang RQ1. Pass "-" only if you intentionally
# want recorded start_t offsets (can be days — not for smoke/CI).
# limit: pass "-" for the full trace.
#
# Env overrides:
#   GPU / PORT / MODEL / CONTROL_PORT / VENV / AGENTREPLAY
#   XPOOL_SATURATION_LOW (sys/sys_lru only, default 0.5)
#
# The control HTTP server (hosting /generate + /flush_cache) listens on
# CONTROL_PORT (default PORT+1). The smg gateway listens on PORT.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARM=${1:?arm}; TRACE=${2:?trace}; STAGGER=${3:?stagger}
MAXCONC=${4:?maxconc}; LIMIT=${5:?limit}; NREPS=${6:?nreps}; OUTDIR=${7:?outdir}

PORT=${PORT:-30100}
GPU=${GPU:-2}
MODEL="${MODEL:-/scratch/yuzhou/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
CONTROL_PORT=${CONTROL_PORT:-$((PORT + 1))}
VENV=${VENV:-/data/yuzhou/projects/tokenspeed/tokenspeed/bin/python}
# Vendored copy — never point this at /data/yuzhou/projects/agentreplay.
AGENTREPLAY=${AGENTREPLAY:-$ROOT/third_party/agentreplay}
XPOOL_SATURATION_LOW=${XPOOL_SATURATION_LOW:-0.5}

# The server JIT-compiles kernels at boot and shells out to `ninja`, which
# lives in the venv's bin/. Callers that launch this script from a shell
# without the venv activated (e.g. setsid from a matrix driver) would
# otherwise hit "FileNotFoundError: 'ninja'" during startup.
export PATH="$(dirname "$VENV"):$PATH"

mkdir -p "$OUTDIR"

# --- OpenMP / affinity workaround (kernel 6.17 sched_setaffinity wedge) -------
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=false
export KMP_AFFINITY=disabled
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=false
export NUMEXPR_NUM_THREADS=1

# smg's Prometheus exporter defaults to a fixed port (8413) regardless of
# --port/--control-port, so running multiple arms in parallel on one host
# collides there even though the gateway/control ports differ. Derive a
# unique one per PORT.
PROMETHEUS_PORT=${PROMETHEUS_PORT:-$((PORT + 8313))}

COMMON=(
  --model "$MODEL"
  --host 127.0.0.1
  --port "$PORT"
  --control-port "$CONTROL_PORT"
  --base-gpu-id 0
  --prometheus-port "$PROMETHEUS_PORT"
)

# Caller-provided extra `ts serve` flags, e.g.
#   EXTRA_COMMON="--max-total-tokens 1200000"
# to cap the KV pool and create genuine, sustained KV scarcity (the regime
# where HiMA's cross-pool machinery pays off — see docs/guides/hima_phase3.md).
# Empty by default so existing callers are unaffected. Word-split on purpose.
EXTRA_COMMON=${EXTRA_COMMON:-}
if [ -n "$EXTRA_COMMON" ]; then
  # shellcheck disable=SC2206
  COMMON+=($EXTRA_COMMON)
fi

# Arm-specific HiMA flags. Mirrors python/tokenspeed/agentreplay/__main__.py
# _PRESETS so `ts serve` can reproduce the same decision-layer behaviour.
case "$ARM" in
  base)
    FLAGS=(
      --radix-eviction-policy lru
    )
    ;;
  sys)
    FLAGS=(
      --radix-eviction-policy lpb
      --lpb-window-s 60.0
      --lpb-hit-deque-maxlen 4096
      --csigma-kv-alpha 1.02e-7
      --csigma-kv-beta 0.0246
      --csigma-kv-gamma 5.97
      --csigma-m 0.0
      --enable-budgeter
      --enable-admitter
      --enable-xpool-dynamic-capacity
      --enable-dynamic-admission-cap
      --budgeter-tick-s 1.0
      --budgeter-pages-per-fire 64
      --xpool-saturation-low "$XPOOL_SATURATION_LOW"
      --xpool-reverse-cooldown-s 2.0
      --xpool-mamba-headroom-slots 0
      --xpool-w-queue 0.25
      --xpool-w-retract 0.20
      --xpool-w-paused 0.0
    )
    ;;
  # sys_ref: the LPB/csigma eviction layer alone, with the crash-prone and
  # throughput-limiting decision-layer components removed:
  #  * --enable-xpool-dynamic-capacity gates newXPoolCappedDrainRetractOperations()
  #    (scheduler.cpp), the proactive-retract path whose alloc_count reconcile
  #    race causes the CUDA illegal-memory-access crash under sustained KV
  #    scarcity (both sys and sys_lru died mid-rep at caps 640k AND 1.1M).
  #  * --enable-admitter / --enable-dynamic-admission-cap systematically held
  #    mean decode concurrency at ~5.8 running-req vs base's ~6.7 while
  #    queue-req sat at 1-2, costing ~15% aggregate throughput for nothing.
  # What remains is the LPB eviction policy + csigma cost model, the component
  # that produced the cache_hit gain (0.73 vs base 0.61).
  sys_ref)
    FLAGS=(
      --radix-eviction-policy lpb
      --lpb-window-s 60.0
      --lpb-hit-deque-maxlen 4096
      --csigma-kv-alpha 1.02e-7
      --csigma-kv-beta 0.0246
      --csigma-kv-gamma 5.97
      --csigma-m 0.0
    )
    ;;
  # sys_dyncap: isolation arm (2026-07-19). LPB eviction + ONLY the XPool
  # dynamic-capacity allocator mode -- no budgeter/admitter/dynamic-cap. Used
  # to attribute the idle retract-churn (sys_hi showed 529 scheduleRetract vs
  # sys_ref's 1 on the same non-binding workload) to the dynamic-capacity
  # allocator path itself rather than the fire/admit decision layer.
  sys_dyncap)
    FLAGS=(
      --radix-eviction-policy lpb
      --lpb-window-s 60.0
      --lpb-hit-deque-maxlen 4096
      --csigma-kv-alpha 1.02e-7
      --csigma-kv-beta 0.0246
      --csigma-kv-gamma 5.97
      --csigma-m 0.0
      --enable-xpool-dynamic-capacity
    )
    ;;
  sys_lru)
    FLAGS=(
      --radix-eviction-policy lru
      --enable-budgeter
      --enable-admitter
      --enable-xpool-dynamic-capacity
      --enable-dynamic-admission-cap
      --budgeter-tick-s 1.0
      --budgeter-pages-per-fire 64
      --xpool-saturation-low "$XPOOL_SATURATION_LOW"
      --xpool-reverse-cooldown-s 2.0
      --xpool-mamba-headroom-slots 0
      --xpool-w-queue 0.25
      --xpool-w-retract 0.20
      --xpool-w-paused 0.0
    )
    ;;
  *)
    echo "[run_arm_ts] unknown arm: $ARM (want base|sys|sys_lru)" >&2
    exit 2
    ;;
esac

echo "[run_arm_ts] arm=$ARM gpu=$GPU port=$PORT control=$CONTROL_PORT prometheus=$PROMETHEUS_PORT"
echo "[run_arm_ts] model=$MODEL"
echo "[run_arm_ts] agentreplay=$AGENTREPLAY"
echo "[run_arm_ts] flags: ${FLAGS[*]}"

kill_port() {
  local p="$1"
  local pid
  pid=$(ss -ltnp 2>/dev/null | grep ":$p " | grep -oE 'pid=[0-9]+' | grep -oE '[0-9]+' | head -1)
  if [ -n "$pid" ]; then
    echo "[run_arm_ts] killing pid=$pid on port $p"
    kill -9 "$pid" 2>/dev/null || true
  fi
}

reap_orphans() {
  # Kill anything still holding PORT / CONTROL_PORT, then any leftover
  # engine/scheduler children for this CUDA_VISIBLE_DEVICES mapping.
  kill_port "$PORT"
  kill_port "$CONTROL_PORT"
  # Best-effort: engine/gateway processes whose cmdline mentions THIS arm's
  # ports. Must stay strictly port-specific: a bare
  # "smg_grpc_servicer.tokenspeed" pattern also matches SIBLING arms' engines
  # and would SIGKILL them when several arms run in parallel (observed: a
  # later-launched arm's reap killed an earlier arm mid-boot -> rc=-9).
  for pid in $(pgrep -f -- "[[:space:]]--port[[:space:]]${PORT}([[:space:]]|\$)|[[:space:]]--control-port[[:space:]]${CONTROL_PORT}([[:space:]]|\$)" 2>/dev/null || true); do
    kill -9 "$pid" 2>/dev/null || true
  done
}

boot_server() {
  # Clear any leftover listeners before boot.
  reap_orphans
  for i in $(seq 1 40); do
    ss -ltn 2>/dev/null | grep -qE ":($PORT|$CONTROL_PORT) " || break
    sleep 1
  done

  # Wait for GPU memory to actually free before rebinding. After a CUDA
  # illegal-memory-access crash the driver can take much longer than a clean
  # exit to tear down the faulted context (observed >20 min stalls on
  # cudaSetDevice/context init in the next boot when a new process attaches
  # to a GPU whose previous occupant's context is still unwinding) -- so
  # also require no compute processes remain attached, not just low memory.
  # 150 x 2s = 300s patience (was 120s memory-only).
  for i in $(seq 1 150); do
    USED=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    NPROC=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    [ "${USED:-0}" -lt 2000 ] 2>/dev/null && [ "${NPROC:-0}" -eq 0 ] 2>/dev/null && break
    sleep 2
  done

  # Boot. CUDA_VISIBLE_DEVICES remaps physical $GPU -> logical 0, so --base-gpu-id 0.
  export CUDA_VISIBLE_DEVICES=$GPU
  export TOKENSPEED_BUDGET_LOG="$OUTDIR/budgeter.jsonl"
  # New process group so teardown can kill the whole tree with kill -9 -$SVPID.
  # Server log is append-mode so restart history is preserved in one file.
  setsid "$VENV" -m tokenspeed.cli serve "${COMMON[@]}" "${FLAGS[@]}" \
    >> "$OUTDIR/server_${ARM}.log" 2>&1 &
  SVPID=$!
  echo "[run_arm_ts] server pgid/pid=$SVPID -> $OUTDIR/server_${ARM}.log"

  # Wait for control /health (engine+gateway+control all up).
  local ready=0
  for i in $(seq 1 240); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
      "http://127.0.0.1:${CONTROL_PORT}/health" 2>/dev/null || true)
    if [ "$code" = "200" ]; then
      ready=1
      echo "[run_arm_ts] ready after ~$((i*5))s (control /health=200)"
      break
    fi
    if ! kill -0 "$SVPID" 2>/dev/null; then
      echo "[run_arm_ts] SERVER DIED DURING BOOT"
      tail -40 "$OUTDIR/server_${ARM}.log"
      return 1
    fi
    sleep 5
  done
  if [ "$ready" != "1" ]; then
    echo "[run_arm_ts] BOOT TIMEOUT"
    tail -40 "$OUTDIR/server_${ARM}.log"
    kill -9 -"$SVPID" 2>/dev/null || true
    reap_orphans
    return 2
  fi
  return 0
}

boot_server_with_retries() {
  # A boot timeout after a crash can mean the just-freed GPU's faulted CUDA
  # context is still unwinding under the driver (see the wait_gpu_free
  # comment in boot_server); one extra teardown+wait+retry cycle recovers
  # this without operator intervention (observed: shifting regime, both
  # arms' final rep aborted the whole arm on a single BOOT TIMEOUT even
  # though the GPU was fully idle a few minutes later).
  local attempts=3 i
  for i in $(seq 1 "$attempts"); do
    if boot_server; then
      return 0
    fi
    echo "[run_arm_ts] boot attempt $i/$attempts failed"
    teardown_server
  done
  return 1
}

teardown_server() {
  kill -9 -"$SVPID" 2>/dev/null || true
  reap_orphans
  sleep 3
}

server_healthy() {
  kill -0 "$SVPID" 2>/dev/null || return 1
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:${CONTROL_PORT}/health" 2>/dev/null || true)
  [ "$code" = "200" ]
}

# Validate a rep JSON: n_ok > 0 and error rate <= MAX_ERR_RATE (default 5%).
# A mid-rep server crash floods the tail of the rep with connection errors,
# so a high error rate marks the rep unusable for stats even though a JSON
# was produced.
MAX_ERR_RATE=${MAX_ERR_RATE:-0.05}
rep_valid() {
  local json="$1"
  [ -f "$json" ] || return 1
  "$VENV" - "$json" "$MAX_ERR_RATE" <<'PYEOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
n_req = d.get("n_requests") or 0
n_ok = d.get("n_ok") or 0
n_err = d.get("n_error") or 0
if n_ok == 0:
    sys.exit(1)
if n_req and n_err / n_req > float(sys.argv[2]):
    sys.exit(1)
sys.exit(0)
PYEOF
}

boot_server || exit 1

LIMARG=(); [ "$LIMIT" != "-" ] && LIMARG=(--limit "$LIMIT")
STAGARG=(); [ "$STAGGER" != "-" ] && STAGARG=(--stagger "$STAGGER")

# Self-healing rep loop: a rep invalidated by a server crash is retried on a
# freshly rebooted server instead of aborting the arm (the old behavior left
# arms with 1-2 valid reps whenever the engine hit a fatal CUDA fault).
# RETRY_BUDGET caps total extra attempts across the whole arm so a
# deterministic crasher cannot loop forever. Crashed attempts are archived as
# ${ARM}_r<rep>.crash<k>.{json,log} for postmortem; the final valid attempt
# owns the canonical ${ARM}_r<rep>.json name that the aggregator reads.
RETRY_BUDGET=${RETRY_BUDGET:-$((NREPS * 2))}
retries_used=0
crashes=0
rep=1
while [ "$rep" -le "$NREPS" ]; do
  if ! server_healthy; then
    echo "[run_arm_ts] server unhealthy before rep $rep -- rebooting"
    tail -20 "$OUTDIR/server_${ARM}.log"
    teardown_server
    if ! boot_server_with_retries; then
      echo "[run_arm_ts] REBOOT FAILED -- aborting arm"
      break
    fi
  fi
  echo "[run_arm_ts] ===== $ARM rep $rep / $NREPS (retries used: $retries_used/$RETRY_BUDGET) ====="
  TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    PYTHONPATH="$AGENTREPLAY${PYTHONPATH:+:$PYTHONPATH}" \
    "$VENV" -m agentreplay replay-tokenspeed \
      --trace "$TRACE" --model "$MODEL" "${STAGARG[@]}" \
      --max-concurrency "$MAXCONC" "${LIMARG[@]}" --flush \
      --url "http://127.0.0.1:${CONTROL_PORT}/generate" \
      --label "${ARM}_r${rep}" \
      --out "$OUTDIR/${ARM}_r${rep}.json" \
      > "$OUTDIR/${ARM}_r${rep}.log" 2>&1
  rc=$?
  if rep_valid "$OUTDIR/${ARM}_r${rep}.json"; then
    echo "[run_arm_ts] rep $rep VALID (rc=$rc): $(grep -oE '\"(cache_hit|throughput_tok_s|n_error|n_ok)\": [^,]+' "$OUTDIR/${ARM}_r${rep}.json" | tr '\n' ' ')"
    rep=$((rep + 1))
    continue
  fi
  # Invalid rep: archive artifacts, decide whether to retry.
  crashes=$((crashes + 1))
  echo "[run_arm_ts] rep $rep INVALID (rc=$rc, crash #$crashes); archiving artifacts"
  [ -f "$OUTDIR/${ARM}_r${rep}.json" ] && mv "$OUTDIR/${ARM}_r${rep}.json" "$OUTDIR/${ARM}_r${rep}.crash${crashes}.json"
  [ -f "$OUTDIR/${ARM}_r${rep}.log" ] && mv "$OUTDIR/${ARM}_r${rep}.log" "$OUTDIR/${ARM}_r${rep}.crash${crashes}.log"
  tail -15 "$OUTDIR/server_${ARM}.log"
  if [ "$retries_used" -ge "$RETRY_BUDGET" ]; then
    echo "[run_arm_ts] RETRY BUDGET EXHAUSTED ($retries_used/$RETRY_BUDGET) -- aborting arm with $((rep - 1)) valid reps"
    break
  fi
  retries_used=$((retries_used + 1))
  echo "[run_arm_ts] retrying rep $rep on a fresh server"
  teardown_server
  if ! boot_server_with_retries; then
    echo "[run_arm_ts] REBOOT FAILED -- aborting arm"
    break
  fi
done
echo "[run_arm_ts] arm summary: valid_reps=$((rep - 1 < NREPS ? rep - 1 : NREPS)) crashes=$crashes retries_used=$retries_used"

# Teardown: kill the process group, then reap any survivors by port.
teardown_server
pgrep -af 'tokenspeed::scheduler|smg_grpc_servicer|smg::router' | grep -v grep || echo "[run_arm_ts] no leftover engine/gateway"
echo "[run_arm_ts] DONE arm=$ARM -> $OUTDIR"
