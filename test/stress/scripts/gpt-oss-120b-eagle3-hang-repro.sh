#!/usr/bin/env bash
# Stress scenario: gpt-oss-120b + EAGLE3, TP2 — decode-wedge HANG REPRO.
#
# This is a regression gate for the TRT-LLM all-reduce-fusion decode wedge.
# It runs the *unmitigated* path on purpose, so:
#
#   * While the bug is OPEN this scenario is EXPECTED TO FAIL — the harness's
#     fatal global-stall detector fires within a couple of minutes (exit 2).
#   * Once the bug is FIXED it should run clean (exit 0) and then guards
#     against regressions.
#
# The wedge is NOT model-specific: the fused norm+allreduce kernel is
# auto-enabled on single-node TP (see server_args: enable_allreduce_fusion is
# auto-on for supported TP configs), and gpt-oss-120b mxfp4 reproduces it at
# just TP2 — a cheap, fast repro (~1 min load, wedges in ~75 s).
#
# Mitigation (the actual off-switch): --comm-fusion-max-num-tokens 0 disables
# the fused kernel (it only fires for num_tokens <= that threshold, default
# 2048). NOTE: --disable-custom-all-reduce does NOT disable this fused path.
#
# Tunables (env): OUT_DIR, DURATION_S (default 600), MAX_CONCURRENCY (default 24).
set -uo pipefail
cd "$(dirname "$0")/../../.."

OUT_DIR="${OUT_DIR:-$PWD/stress-out}"
DURATION_S="${DURATION_S:-600}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-24}"
mkdir -p "$OUT_DIR"

# Custom all-reduce + fused allreduce left ON (auto-enabled for single-node TP2).
SERVE="tokenspeed serve \
  --model openai/gpt-oss-120b \
  --trust-remote-code \
  --host 127.0.0.1 --port 8000 \
  --attn-tp-size 2 --moe-tp-size 2 \
  --max-model-len 80000 --max-num-seqs 24 \
  --gpu-memory-utilization 0.9 \
  --attention-backend trtllm --moe-backend flashinfer_mxfp4 \
  --reasoning-parser base \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path nvidia/gpt-oss-120b-Eagle3-long-context \
  --speculative-num-steps 3 \
  --enable-cache-report --enable-metrics"

# Text-only reality_mix (gpt-oss has no vision). grammar disabled (irrelevant to
# the wedge; avoids gpt-oss structured-output noise). Prompts capped to fit the
# 80k context; aggressive long-decode mirrors the regime that surfaces the wedge.
python3 -m test.stress run \
  --launch-cmd "$SERVE" \
  --launch-timeout 1800 \
  --base-url http://127.0.0.1:8000 \
  --model openai/gpt-oss-120b \
  --workload reality_mix \
  --workload-arg grammar_fraction=0 \
  --workload-arg cancel_fraction=0.15 \
  --workload-arg cached_fraction=0.5 \
  --workload-arg prompt_tokens_max=40000 \
  --workload-arg max_tokens_cap=32768 \
  --workload-arg very_long_weight=20 \
  --arrival sawtooth --min-concurrency 1 \
  --max-concurrency "$MAX_CONCURRENCY" --triangle-period 180 \
  --duration "$DURATION_S" --request-timeout 1200 \
  --stall-timeout 20 --global-stall-timeout 20 \
  --metrics-interval 10 --accept-len-min 1.1 \
  --out "$OUT_DIR"
rc=$?

if [ -f "$OUT_DIR/events.jsonl" ]; then
  python3 -m test.stress summarize --events "$OUT_DIR/events.jsonl" \
    | tee "$OUT_DIR/summary.txt" || true
fi

exit "$rc"
