"""HiMA e2e inference test with Qwen3.5-35B-A3B (hybrid MoE model).

Validates:
  1. LPB eviction: prefix cache hits with lpb policy
  2. BudgetAgent: enable_budgeter + enable_admitter does not crash
  3. LPB vs LRU: cached_tokens comparison under identical request sequence

Usage:
    conda activate AgentServe
    export PYTHONPATH=python:tokenspeed-scheduler/python
    cd /home/songyang/projects/tokenspeed
    python test/test_e2e_hima_infer.py
"""

import time

import torch
from tokenspeed.runtime.entrypoints.engine import Engine

MODEL = "/home/songyang/models/Qwen3.5-35B-A3B"

COMMON_KWARGS = dict(
    model=MODEL,
    dtype="bfloat16",
    # "flashinfer" uses TRT-LLM FMHA runner which requires HPC Blackwell (sm100/sm103).
    # On consumer Blackwell (sm120) use "triton" which has no arch restriction.
    attention_backend="triton",
    moe_backend="auto",
    mamba_full_memory_ratio=0.15,
    base_gpu_id=1,
)

SHARED_PREFIX = "The quick brown fox jumps over the lazy dog. " * 100  # ~700 tokens


def _make_engine(policy: str, enable_budgeter: bool = False) -> Engine:
    return Engine(
        **COMMON_KWARGS,
        # HiMA Phase 1
        radix_eviction_policy=policy,
        lpb_window_s=30.0,
        lpb_hit_deque_maxlen=1024,
        csigma_kv_alpha=1.02e-7,
        csigma_kv_beta=0.0246,
        csigma_kv_gamma=5.97,
        csigma_m=0.0,
        # HiMA Phase 2
        enable_budgeter=enable_budgeter,
        enable_admitter=enable_budgeter,
        budgeter_tick_s=1.0,
    )


def _teardown(engine: Engine) -> None:
    """Fully release an engine's GPU memory before the next one is created.

    The 35B model needs ~86 GB; a single GPU only fits one engine at a time,
    so each engine's scheduler subprocess must be killed (not just `del`ed)
    and the memory returned to the OS before the next engine is constructed.
    """
    engine.shutdown()
    # kill_process_tree is synchronous, but the driver reclaims device memory
    # asynchronously; give it a moment so the next engine sees a free GPU.
    time.sleep(5)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_lpb_prefix_cache_hits():
    print("\n[Test 1] LPB prefix cache hit rate ...")
    engine = _make_engine(policy="lpb")
    try:
        prompts = [SHARED_PREFIX + f" question {i}?" for i in range(6)]

        for p in prompts:
            engine.generate(p, sampling_params={"max_new_tokens": 4})

        total_cached = 0
        for i, p in enumerate(prompts):
            resp = engine.generate(p, sampling_params={"max_new_tokens": 4})
            cached = resp["meta_info"].get("cached_tokens", 0)
            total_cached += cached
            print(f"  prompt[{i}] cached_tokens={cached}")

        assert total_cached > 0, "LPB should have prefix cache hits"
        print(f"  ✓ total cached_tokens={total_cached}")
    finally:
        _teardown(engine)


def test_budgeter_does_not_crash():
    print("\n[Test 2] BudgetAgent stability ...")
    engine = _make_engine(policy="lpb", enable_budgeter=True)
    try:
        for i in range(8):
            resp = engine.generate(
                SHARED_PREFIX + f" answer {i}",
                sampling_params={"max_new_tokens": 16},
            )
            print(
                f"  step[{i}] tokens="
                f"{resp['meta_info'].get('completion_tokens', '?')}"
            )

        print("  ✓ BudgetAgent: no crash")
    finally:
        _teardown(engine)


def test_lpb_vs_lru_cache_hits():
    print("\n[Test 3] LPB vs LRU cache hit comparison ...")
    results = {}
    for policy in ("lru", "lpb"):
        engine = _make_engine(policy=policy)
        try:
            prompts = [SHARED_PREFIX + f" item {i}" for i in range(8)]

            for p in prompts:
                engine.generate(p, sampling_params={"max_new_tokens": 4})

            hits = sum(
                engine.generate(p, sampling_params={"max_new_tokens": 4})[
                    "meta_info"
                ].get("cached_tokens", 0)
                for p in prompts
            )
            results[policy] = hits
            print(f"  [{policy}] total cached_tokens={hits}")
        finally:
            _teardown(engine)

    assert results["lpb"] >= results["lru"], (
        f"LPB ({results['lpb']}) should be >= LRU ({results['lru']})"
    )
    print("  ✓ LPB >= LRU")


if __name__ == "__main__":
    test_lpb_prefix_cache_hits()
    test_budgeter_does_not_crash()
    test_lpb_vs_lru_cache_hits()
    print("\n===== All HiMA e2e inference tests PASSED =====")
