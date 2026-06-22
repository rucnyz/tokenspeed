"""HiMA Phase 2 XPool bi-directional stress test.

Drives capacity transfers in **both** directions:

* ``kv_to_mamba``:  mamba pressure >> KV pressure (many unique short prompts)
* ``mamba_to_kv``:  KV pressure >> mamba pressure (long shared-prefix prompts
  submitted concurrently to keep KV pages allocated while mamba pressure drops)

Strategy
--------
Phase A  – Warm up with mamba-heavy requests so the budgeter fires kv→mamba.
Phase B  – Submit long-prefix KV-heavy requests concurrently.  Concurrent
           submission is needed so that many KV pages stay allocated at the
           same time; sequential requests free pages before the next starts.
           With KV pressure high and mamba pressure having dropped (previous
           kv→mamba fires freed KV → mamba util falls), the budgeter fires
           mamba→KV.
Phase C  – Repeat in the other direction to confirm both paths work.

Verification strategy
---------------------
* Output correctness: all outputs non-empty.
* Transfer coverage: grep log for both ``kv_to_mamba`` and ``mamba_to_kv``
  committed fires.

Run with::

    cd /home/songyang/projects/tokenspeed
    PYTHONPATH=python TOKENSPEED_MAMBA_SSM_DTYPE=float32 \\
        python test/test_stress_xpool.py 2>&1 | tee /tmp/stress_xpool.log

Verify directions after run::

    grep -E "fire committed.*kv_to_mamba" /tmp/stress_xpool.log | wc -l
    grep -E "fire committed.*mamba_to_kv" /tmp/stress_xpool.log | wc -l
"""
from __future__ import annotations

import threading
import time

MODEL = "/home/songyang/models/Qwen3.5-35B-A3B"


# ---------------------------------------------------------------------------
# Prompt generators
# ---------------------------------------------------------------------------

def _kv_heavy_prompts(n: int, salt: int = 0) -> list[str]:
    """Long UNIQUE prompts so each request allocates many fresh KV pages.

    With page_size=64 and the hybrid model's tiny mamba pool (~37 slots) vs
    huge KV pool (~12k pages), each request must contribute a lot of KV
    pressure to overcome mamba pressure: kv_util needs to exceed
    mamba_util(eff).  We therefore make every prompt ~7K tokens of UNIQUE
    text (the salt+index prefix breaks shared-prefix dedup, so each request
    really does allocate its own KV pages near max_prefill_tokens=8192).
    """
    base = (
        "This is a detailed technical document about distributed systems, "
        "consensus algorithms, fault-tolerant storage, and the intricate "
        "dance of bytes flowing through wire protocols designed by tired "
        "engineers in the small hours of long winter nights with coffee. "
        * 350  # ~7K tokens after tokenisation
    )
    return [
        f"[salt={salt} q={i} unique={salt * 1000 + i}] {base} "
        f"Now, summarise the key insights in one short paragraph."
        for i in range(n)
    ]


def _mamba_heavy_prompts(n: int, salt: int = 0) -> list[str]:
    """Short unique prompts → unique mamba slots, low KV pressure."""
    return [
        f"[s{salt}] Prompt {i}: Write a haiku about {i * 7 + 3 + salt}."
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_sequential(
    engine,
    prompts: list[str],
    tag: str,
    results: list[str],
    corruption_count_ref: list[int],
    max_new_tokens: int = 48,
) -> None:
    """Submit prompts one at a time and collect outputs."""
    print(f"[{tag}] sending {len(prompts)} prompts sequentially …")
    for i, p in enumerate(prompts):
        out = engine.generate(p, sampling_params={"max_new_tokens": max_new_tokens})
        text = str(out)
        results.append(text)
        if not text.strip():
            corruption_count_ref[0] += 1
            print(f"  !! EMPTY output at {tag}[{i}]")
        else:
            print(f"  [{tag}][{i}] ok — {text[:50]!r}")


def _run_concurrent(
    engine,
    prompts: list[str],
    tag: str,
    results: list[str],
    corruption_count_ref: list[int],
    max_new_tokens: int = 48,
) -> None:
    """Submit prompts from N threads simultaneously.

    Concurrent submission keeps pages/slots allocated at the same time,
    which drives utilisation high enough for the budgeter to react.
    Using more output tokens (max_new_tokens) ensures the KV pages stay
    allocated across multiple budget-tick intervals (tick_s=0.5).
    """
    print(f"[{tag}] sending {len(prompts)} prompts concurrently "
          f"(max_new_tokens={max_new_tokens}) …")
    lock = threading.Lock()

    def _submit(idx: int, prompt: str) -> None:
        out = engine.generate(
            prompt, sampling_params={"max_new_tokens": max_new_tokens}
        )
        text = str(out)
        with lock:
            results.append(text)
            if not text.strip():
                corruption_count_ref[0] += 1
                print(f"  !! EMPTY output at {tag}[{idx}]")
            else:
                print(f"  [{tag}][{idx}] ok — {text[:50]!r}")

    threads = [
        threading.Thread(target=_submit, args=(i, p), daemon=True)
        for i, p in enumerate(prompts)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from tokenspeed.runtime.entrypoints.engine import Engine

    print("Creating Engine with xpool dynamic capacity enabled …")
    engine = Engine(
        model=MODEL,
        dtype="bfloat16",
        attention_backend="triton",
        moe_backend="auto",
        mamba_full_memory_ratio=0.15,
        base_gpu_id=1,
        log_level="info",
        # HiMA Phase 2
        enable_budgeter=True,
        enable_admitter=True,
        enable_xpool_dynamic_capacity=True,
        budgeter_pages_per_fire=32,
        budgeter_tick_s=0.5,
        # Small margin + fast EWMA: any utilisation imbalance immediately
        # triggers a fire so the mock workload reliably drives both directions.
        # Production uses xpool_nb_margin=0.05 and xpool_ewma_tau_s=1.0 to
        # avoid thrashing; the test uses aggressive values for determinism.
        xpool_nb_margin=0.001,
        xpool_ewma_tau_s=0.001,
    )
    print("Engine ready.\n")

    results: list[str] = []
    corruption_count_ref: list[int] = [0]

    # ------------------------------------------------------------------
    # Phase A: mamba-heavy — drives kv_to_mamba transfers.
    # Run 5 rounds so the mamba prefix cache accumulates ~25+ evictable
    # states.  These cached states are what later lets Phase B's KV-heavy
    # requests temporarily inflate kv_util without inflating mamba_util(eff):
    # the budgeter subtracts evictable slots from mamba_used in the effective
    # util computation, so cached-but-evictable slots no longer block the
    # mamba_to_kv direction.
    # ------------------------------------------------------------------
    for r in range(5):
        _run_sequential(
            engine, _mamba_heavy_prompts(8, salt=r * 100),
            f"mamba_heavy_r{r}", results, corruption_count_ref,
        )
        time.sleep(1.5)  # let budgeter tick

    # Allow ewma_mamba to fully decay and all active mamba slots to release
    # before Phase B.  After this sleep the snapshot should look like
    # mamba_used≈0, mamba_evict≈25+ (cached states retained).
    print("[phase_transition] sleeping 2 s to let mamba_used drop to 0 …")
    time.sleep(2.0)

    # ------------------------------------------------------------------
    # Phase B: KV-heavy concurrent — drives mamba_to_kv transfers.
    # Empirical observation from BudgetTick logs: each request consumes ~4
    # mamba slots (one per mamba layer), not 1.  With mamba_total=37 and
    # Phase A leaving ~12 evictable mamba slots, the sweet spot is:
    #   * 3 concurrent: mamba_used ≈ 3 × 4 = 12 ≤ mamba_evict=12,
    #     so mamba_util(eff) = max(0, used - evict)/total = 0.
    #   * Each prompt ~7K tokens (unique) so each request allocates ~240
    #     KV pages → 3 × 240 = ~720 KV pages active, kv_util ≈ 6%.
    #   * 6% > 0% + 0.001 margin → budgeter fires mamba_to_kv repeatedly.
    # 4-concurrent (previous setting) pushes mamba_used to 16 which exceeds
    # mamba_evict=12 and inflates mamba_util(eff) to ~11%, exceeding the
    # achievable kv_util ceiling for hybrid models with this pool ratio.
    # ------------------------------------------------------------------
    for r in range(3):
        _run_concurrent(
            engine, _kv_heavy_prompts(3, salt=500 + r * 100),
            f"kv_heavy_r{r}", results, corruption_count_ref,
            max_new_tokens=512,
        )
        time.sleep(1.5)  # let budgeter tick

    # ------------------------------------------------------------------
    # Phase C: alternate again to confirm both paths repeat
    # ------------------------------------------------------------------
    _run_sequential(
        engine, _mamba_heavy_prompts(8, salt=900),
        "mamba_heavy_final", results, corruption_count_ref,
    )
    time.sleep(1.5)
    _run_concurrent(
        engine, _kv_heavy_prompts(3, salt=950),
        "kv_heavy_final", results, corruption_count_ref,
        max_new_tokens=512,
    )
    time.sleep(2.0)

    engine.shutdown()
    time.sleep(3)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Total outputs : {len(results)}")
    print(f"Empty outputs : {corruption_count_ref[0]}")

    if corruption_count_ref[0] > 0:
        print("FAIL: data corruption detected (empty outputs).")
        raise SystemExit(1)

    print(
        "PASS: all outputs non-empty.\n\n"
        "Verify bidirectional fires in log:\n"
        "  grep -c 'XPool fire committed.*kv_to_mamba' /tmp/stress_bidirectional.log\n"
        "  grep -c 'XPool fire committed.*mamba_to_kv' /tmp/stress_bidirectional.log\n"
    )
    print("=== stress xpool PASS ===")


if __name__ == "__main__":
    main()
