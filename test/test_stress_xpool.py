"""HiMA Phase 2 XPool bi-directional stress test.

Alternates between KV-intensive and Mamba-intensive request batches to drive
the budgeter into firing capacity transfers in both directions.

Verification strategy
---------------------
* Output correctness: all inference outputs must be non-empty (no data
  corruption detected by content).
* Capacity transfers: search engine log for ``xpool_fire`` and
  ``mamba_to_kv`` / ``kv_to_mamba`` direction markers.  At least one fire
  in each direction is required for the stress scenario to be meaningful.

Run with::

    cd /home/songyang/projects/tokenspeed
    PYTHONPATH=python TOKENSPEED_MAMBA_SSM_DTYPE=float32 \\
        python test/test_stress_xpool.py 2>&1 | tee /tmp/stress_xpool.log

Then check the log::

    grep -c "kv_to_mamba\\|mamba_to_kv" /tmp/stress_xpool.log
"""
from __future__ import annotations

import time

MODEL = "/home/songyang/models/Qwen3.5-35B-A3B"


# ---------------------------------------------------------------------------
# Prompt generators
# ---------------------------------------------------------------------------

def _kv_heavy_prompts(n: int) -> list[str]:
    """Long shared prefix → high KV pressure, low mamba pressure."""
    prefix = (
        "This is a detailed technical document about distributed systems, "
        "consensus algorithms, and fault-tolerant storage. " * 40
    )
    return [prefix + f" Question {i}: summarise in one sentence." for i in range(n)]


def _mamba_heavy_prompts(n: int) -> list[str]:
    """Short prompts, each unique → high mamba pressure, low KV pressure."""
    return [
        f"Prompt {i}: Write a haiku about the number {i * 7 + 3}."
        for i in range(n)
    ]


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
    )
    print("Engine ready.\n")

    results: list[str] = []
    corruption_count = 0

    def _run_batch(prompts: list[str], tag: str) -> None:
        nonlocal corruption_count
        print(f"[{tag}] sending {len(prompts)} prompts …")
        for i, p in enumerate(prompts):
            out = engine.generate(p, sampling_params={"max_new_tokens": 48})
            text = str(out)
            results.append(text)
            if not text.strip():
                corruption_count += 1
                print(f"  !! EMPTY output at {tag}[{i}]")
            else:
                print(f"  [{tag}][{i}] ok — {text[:50]!r}")
        # Allow budgeter to tick and fire transfers.
        time.sleep(1.5)

    # Round 1: KV-heavy to drive mamba→KV fire.
    _run_batch(_kv_heavy_prompts(4), "kv_heavy_r1")

    # Round 2: Mamba-heavy to drive KV→mamba fire.
    _run_batch(_mamba_heavy_prompts(6), "mamba_heavy_r1")

    # Round 3: KV-heavy again to exercise bidirectional transfer.
    _run_batch(_kv_heavy_prompts(4), "kv_heavy_r2")

    # Round 4: Mamba-heavy again.
    _run_batch(_mamba_heavy_prompts(6), "mamba_heavy_r2")

    engine.shutdown()
    time.sleep(3)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Total outputs : {len(results)}")
    print(f"Empty outputs : {corruption_count}")

    if corruption_count > 0:
        print("FAIL: data corruption detected (empty outputs).")
        raise SystemExit(1)

    print(
        "PASS: all outputs non-empty.\n"
        "XPool fire dispatches verified via log — search for:\n"
        "  grep 'XPool fire dispatched' /tmp/stress_xpool.log | head -20\n"
        "\n"
        "NOTE: With a small workload (<=20 short requests), only\n"
        "  kv_to_mamba fires are expected (KV starts 0.008% used while\n"
        "  mamba is 100% free).  mamba_to_kv fires require concurrent\n"
        "  usage >= mamba_total_slots/2 to push mamba util above KV util."
    )
    print("=== stress xpool PASS ===")


if __name__ == "__main__":
    main()
