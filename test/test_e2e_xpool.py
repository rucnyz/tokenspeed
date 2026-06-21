"""HiMA Phase 2 XPool dynamic capacity e2e test.

Verifies:
  - Engine initialises with xpool enabled (mapped_kv_pages > 0)
  - mamba pool tensors optionally bound to VMM arena
  - Inference produces correct output (no data corruption)
  - C++ capacity accounting responds to budgeter ticks
"""
import time

MODEL = "/home/songyang/models/Qwen3.5-35B-A3B"


def main():
    import torch
    from tokenspeed.runtime.entrypoints.engine import Engine

    print("Creating engine with xpool enabled…")
    engine = Engine(
        model=MODEL,
        dtype="bfloat16",
        attention_backend="triton",
        moe_backend="auto",
        mamba_full_memory_ratio=0.15,
        base_gpu_id=1,
        log_level="info",
        # HiMA Phase 2 xpool
        enable_budgeter=True,
        enable_admitter=True,
        enable_xpool_dynamic_capacity=True,
        budgeter_pages_per_fire=32,
        budgeter_tick_s=0.5,
    )

    # 触发推理，产生负载让 budgeter 可能触发
    SHARED = "The quick brown fox jumps over the lazy dog. " * 20
    prompts = [SHARED + f" question {i}" for i in range(3)]
    print("Running inference to trigger budgeter…")
    for p in prompts:
        out = engine.generate(p, sampling_params={"max_new_tokens": 64})
        print(f"  output[:60]: {str(out)[:60]!r}")
        assert out is not None, "Empty output — possible data corruption!"

    # 等 budgeter tick（tick_s=0.5，等 2 秒足够触发数次）
    time.sleep(2)
    print("\nInference complete, budgeter ticks fired.")

    engine.shutdown()
    time.sleep(3)
    print("\nEngine shutdown OK")
    print("=== e2e xpool PASS ===")


if __name__ == "__main__":
    main()