#!/usr/bin/env bash
# Copyright (c) 2026 LightSeek Foundation
# Licensed under the same terms as the rest of tokenspeed (see LICENSE).
#
# HiMA Phase 3 A/B/C benchmark driver. Runs the agentreplay harness against
# a real Claude-Code trace, sweeping one preset per "arm", with N reps each.
#
# Default arms: base (LRU only) + sys (LPB + HiMA decision layer).
# Set ARMS="base sys sys_lru" for a three-arm sweep that isolates the
# LPB-vs-LRU contribution from the HiMA decision layer.
#
# Usage:
#   bash scripts/bench_hima.sh <run_id> [<reps>]
#
# Required environment overrides (defaults shown):
#   MODEL=/home/songyang/models/Qwen3.5-9B
#   TRACE=dataset/claude-code-traces/traces/cc_qwen_mamba.jsonl
#   MAX_SESSIONS=50
#   TIME_SCALE=1.0
#   MAX_INTER_SESSION_GAP_S=5.0
#   GPU_ID=1            # index of the first GPU to use (never 0 by default)
#   N_PARALLEL_GPUS=0   # >0: run arms in parallel, one arm per GPU starting at GPU_ID
#                       #     automatically sets PARALLEL_ARMS=1 and derives GPU_ID_STEP=1
#                       #     arms are assigned GPU_ID, GPU_ID+1, ..., GPU_ID+N-1
#                       #     if #arms > N, extra arms wrap around (round-robin)
#   ARMS="base sys"
#   PARALLEL_ARMS=0     # set to 1 manually (or use N_PARALLEL_GPUS which implies it)
#
# Parallel mode (PARALLEL_ARMS=1 or N_PARALLEL_GPUS>0):
#   Arms run concurrently. Arm i uses GPU (GPU_ID + i % N_PARALLEL_GPUS) when
#   N_PARALLEL_GPUS is set, else GPU_ID + i*GPU_ID_STEP. Each arm gets a unique
#   port (BASE_PORT + i*PORT_STRIDE) to avoid ZMQ conflicts. All arms within a
#   rep must finish before the next rep and the compare step begin.
#
# Produces:
#   runs/<run_id>/<arm>/rep<N>/{per_request.jsonl,summary.json,budget.jsonl,config.json}
#   Plus a final compare table for each (sys arm) vs base printed to stdout.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${1:-${USER}_$(date +%Y%m%d_%H%M%S)}"
REPS="${2:-1}"
MODEL="${MODEL:-/home/songyang/models/Qwen3.5-9B}"
TRACE="${TRACE:-dataset/claude-code-traces/traces/cc_qwen_mamba.jsonl}"
MAX_SESSIONS="${MAX_SESSIONS:-50}"
# Skip steps with prompts longer than the model's context window to avoid
# ValueError from the engine (cc_qwen traces contain multi-turn sessions
# that grow past 262K tokens).  Default matches Qwen3.5-9B max_model_len.
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-262000}"
TIME_SCALE="${TIME_SCALE:-1.0}"
# The cc_qwen* traces use absolute wall-clock t spanning weeks of real
# Claude Code usage. Without a cap we'd sleep for hours between sessions.
# 5s is short enough to keep the benchmark under ~10min but long enough
# to let each session settle in/out of cache before the next one starts.
MAX_INTER_SESSION_GAP_S="${MAX_INTER_SESSION_GAP_S:-5.0}"
GPU_ID="${GPU_ID:-1}"
GPU_ID_STEP="${GPU_ID_STEP:-1}"
N_PARALLEL_GPUS="${N_PARALLEL_GPUS:-0}"
PARALLEL_ARMS="${PARALLEL_ARMS:-0}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
ARMS="${ARMS:-base sys}"

# N_PARALLEL_GPUS is the user-friendly knob: it implies PARALLEL_ARMS=1.
if [[ "$N_PARALLEL_GPUS" -gt 0 ]]; then
    PARALLEL_ARMS=1
fi

OUT_BASE="runs/${RUN_ID}"
mkdir -p "$OUT_BASE"

# Activate the same env as the rest of the repo. We don't fail hard if the
# user is already in an env that has tokenspeed installed -- this is mostly
# for shells that invoke the script fresh.
if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
    if [[ -f /home/songyang/miniconda3/etc/profile.d/conda.sh ]]; then
        # shellcheck disable=SC1091
        source /home/songyang/miniconda3/etc/profile.d/conda.sh
        conda activate AgentServe
    fi
fi

# Always prepend the conda env bin dir so JIT build tools (e.g. flashinfer
# calls `ninja` via subprocess) are found regardless of how the caller set up
# PATH.  This is idempotent if AgentServe is already active.
_CONDA_ENV_BIN="/home/songyang/miniconda3/envs/AgentServe/bin"
if [[ -d "$_CONDA_ENV_BIN" ]]; then
    export PATH="${_CONDA_ENV_BIN}:${PATH}"
fi

export PYTHONPATH="python:tokenspeed-scheduler/python:${PYTHONPATH:-}"

BASE_PORT="${BASE_PORT:-8100}"
PORT_STRIDE="${PORT_STRIDE:-100}"

run_one_arm() {
    local arm="$1"; local rep="$2"; local gpu="$3"; local port="$4"
    local out_dir="$OUT_BASE/$arm/rep${rep}"
    # Skip if a prior run already produced a complete summary.json. This makes
    # bench restart-friendly after a single arm crashes: just `rm -rf` that
    # arm's rep dir and rerun the script.
    if [[ -s "$out_dir/summary.json" ]]; then
        echo ""
        echo "==> ${arm}/rep${rep} already complete; skipping (rm to redo)"
        return 0
    fi
    mkdir -p "$out_dir"
    echo ""
    echo "==> Running ${arm} arm rep ${rep} on GPU ${gpu} port ${port} -> ${out_dir}"
    # We forward EXTRA_OVERRIDES verbatim so users can layer per-run knobs
    # (e.g. EXTRA_OVERRIDES="--override xpool_nb_margin=0.03").
    # shellcheck disable=SC2086
    python -m tokenspeed.agentreplay \
        --trace "$TRACE" \
        --model "$MODEL" \
        --preset "$arm" \
        --output-dir "$out_dir" \
        --max-sessions "$MAX_SESSIONS" \
        --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
        --time-scale "$TIME_SCALE" \
        --max-inter-session-gap-s "$MAX_INTER_SESSION_GAP_S" \
        --base-gpu-id "$gpu" \
        --override "port=${port}" \
        $EXTRA_OVERRIDES \
        2>&1 | tee "${out_dir}/run.log"
}

for rep in $(seq 0 $((REPS-1))); do
    if [[ "$PARALLEL_ARMS" == "1" ]]; then
        # Launch every arm concurrently.
        # GPU assignment:
        #   N_PARALLEL_GPUS>0 → arm i uses GPU_ID + (i % N_PARALLEL_GPUS)
        #   otherwise         → arm i uses GPU_ID + i*GPU_ID_STEP
        # Port: BASE_PORT + i*PORT_STRIDE (always unique per arm index)
        arm_idx=0
        declare -a _pids=()
        for arm in $ARMS; do
            if [[ "$N_PARALLEL_GPUS" -gt 0 ]]; then
                arm_gpu=$(( GPU_ID + arm_idx % N_PARALLEL_GPUS ))
            else
                arm_gpu=$(( GPU_ID + arm_idx * GPU_ID_STEP ))
            fi
            arm_port=$(( BASE_PORT + arm_idx * PORT_STRIDE ))
            run_one_arm "$arm" "$rep" "$arm_gpu" "$arm_port" &
            _pids+=($!)
            arm_idx=$(( arm_idx + 1 ))
        done
        echo ""
        echo "==> Waiting for all arms (rep ${rep}) to finish ..."
        for pid in "${_pids[@]}"; do
            wait "$pid" || { echo "ERROR: arm pid=${pid} failed" >&2; exit 1; }
        done
        echo "==> All arms rep ${rep} done."
        unset _pids
    else
        arm_idx=0
        for arm in $ARMS; do
            arm_port=$(( BASE_PORT + arm_idx * PORT_STRIDE ))
            run_one_arm "$arm" "$rep" "$GPU_ID" "$arm_port"
            arm_idx=$(( arm_idx + 1 ))
        done
    fi
done

# Compare every non-`base` arm against base. Skip if there's no base arm.
if [[ " $ARMS " == *" base "* ]]; then
    for arm in $ARMS; do
        if [[ "$arm" == "base" ]]; then continue; fi
        echo ""
        echo "==> Comparing ${OUT_BASE}/base vs ${OUT_BASE}/${arm}"
        python -m tokenspeed.agentreplay.compare \
            "${OUT_BASE}/base" "${OUT_BASE}/${arm}" \
            --base-name "base(LRU)" --sys-name "${arm}"
    done
fi
