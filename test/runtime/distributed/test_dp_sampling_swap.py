"""Tests for tokenspeed.runtime.distributed.dp_sampling_swap.

Spawns real NCCL workers and verifies that ``swap_batch_vocab`` re-shards
logits exactly the way the Batch-DP spec-sampling pipeline expects:

  rank r before: [pad_bs * N, V/TP]  -- vocab cols [r*V/TP, (r+1)*V/TP)
  rank r after : [K_req * N,  V]     -- request rows [r*K_req, (r+1)*K_req)

The ground-truth tensor for the test is a globally-ordered ``arange`` so
every cell encodes (global request, draft position, vocab column); after
the swap each rank's output must equal the matching slice of that
ground-truth tensor.

Usage:
    python -m pytest test/runtime/distributed/test_dp_sampling_swap.py -v
"""

import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _worker_main(rank, world_size, port, test_fn, error_dict, args):
    try:
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
        )

        from tokenspeed.runtime.distributed.process_group_manager import (
            process_group_manager as pg_manager,
        )

        group = tuple(range(world_size))
        pg_manager.init_process_group(group)

        test_fn(rank=rank, world_size=world_size, device=device, group=group, **args)

        dist.destroy_process_group()
    except Exception:
        import traceback

        error_dict[rank] = traceback.format_exc()


def _run(world_size, test_fn, **args):
    if world_size > torch.cuda.device_count():
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")
    port = _get_open_port()
    error_dict = mp.Manager().dict()
    mp.spawn(
        _worker_main,
        args=(world_size, port, test_fn, error_dict, args),
        nprocs=world_size,
        join=True,
    )
    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))


# ---------------------------------------------------------------------------
# Worker test bodies
# ---------------------------------------------------------------------------


def _ground_truth_full(pad_bs: int, n: int, vocab: int, *, dtype, device):
    """[pad_bs * N, V] where cell (i, v) = i * V + v (compact globally-unique
    ids). Each rank then takes a vocab-column slice as its local_logits."""
    return torch.arange(pad_bs * n * vocab, dtype=dtype, device=device).view(
        pad_bs * n, vocab
    )


def _test_swap_matches_reference(
    rank, world_size, device, group, *, pad_bs, n, vocab, dtype
):
    from tokenspeed.runtime.distributed.dp_sampling_swap import swap_batch_vocab

    tp = world_size
    v_local = vocab // tp
    k_req = pad_bs // tp

    full = _ground_truth_full(pad_bs, n, vocab, dtype=dtype, device=device)

    # rank r owns vocab cols [r*v_local, (r+1)*v_local) — exactly today's
    # vocab-parallel LM head shard layout.
    local_logits = full[:, rank * v_local : (rank + 1) * v_local].contiguous()

    out = swap_batch_vocab(
        local_logits,
        tp_size=tp,
        pad_bs=pad_bs,
        num_tokens_per_req=n,
        vocab_size=vocab,
        rank=rank,
        group=group,
    )

    expected = full[rank * k_req * n : (rank + 1) * k_req * n].contiguous()
    assert tuple(out.shape) == tuple(
        expected.shape
    ), f"shape mismatch: got {tuple(out.shape)} expected {tuple(expected.shape)}"
    torch.testing.assert_close(out, expected)


def _test_swap_chain_safety(
    rank, world_size, device, group, *, pad_bs, n, vocab, dtype
):
    """Per-request N rows must end up on the same rank, in order. We tag
    each cell with (global_req, draft_pos) and verify rank r's output owns
    exactly requests [r*K_req, (r+1)*K_req) with draft positions 0..N-1
    contiguous per request.
    """
    from tokenspeed.runtime.distributed.dp_sampling_swap import swap_batch_vocab

    tp = world_size
    v_local = vocab // tp
    k_req = pad_bs // tp

    # cell value = global_req * 10_000 + draft_pos * 100 + vocab_col.
    # 10_000 is well above any (n * 100 + vocab_col) at the sizes we test,
    # so collisions are impossible.
    full = torch.empty(pad_bs * n, vocab, dtype=dtype, device=device)
    for req in range(pad_bs):
        for d in range(n):
            base = req * 10_000 + d * 100
            full[req * n + d] = torch.arange(vocab, dtype=dtype, device=device) + base
    local_logits = full[:, rank * v_local : (rank + 1) * v_local].contiguous()

    out = swap_batch_vocab(
        local_logits,
        tp_size=tp,
        pad_bs=pad_bs,
        num_tokens_per_req=n,
        vocab_size=vocab,
        rank=rank,
        group=group,
    )

    for r_local in range(k_req):
        global_req = rank * k_req + r_local
        for d in range(n):
            row = out[r_local * n + d]
            expected_first = global_req * 10_000 + d * 100  # vocab_col == 0
            assert int(row[0].item()) == expected_first, (
                f"rank={rank} r_local={r_local} d={d} got row[0]={int(row[0].item())}"
                f" expected {expected_first}"
            )
            # Last vocab col should differ by (vocab-1) from the first.
            assert int(row[-1].item()) == expected_first + (vocab - 1)


# ---------------------------------------------------------------------------
# Public test cases
# ---------------------------------------------------------------------------

WORLD_SIZES = [
    pytest.param(2, id="tp2"),
    pytest.param(4, id="tp4"),
]

# (pad_bs, N, vocab) — all chosen so pad_bs is divisible by every world_size
# in WORLD_SIZES and vocab stays small for runtime budget.
SHAPES = [
    pytest.param(8, 1, 64, id="sample_pad_bs8"),
    pytest.param(8, 4, 64, id="spec_pad_bs8_n4"),
    pytest.param(12, 3, 96, id="spec_pad_bs12_n3"),
]

DTYPES = [
    pytest.param(torch.float32, id="fp32"),
    pytest.param(torch.bfloat16, id="bf16"),
]


class TestDPSamplingSwap:

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    @pytest.mark.parametrize("pad_bs,n,vocab", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_swap_matches_reference(self, world_size, pad_bs, n, vocab, dtype):
        if pad_bs % world_size != 0:
            pytest.skip(f"pad_bs={pad_bs} not divisible by tp={world_size}")
        if vocab % world_size != 0:
            pytest.skip(f"vocab={vocab} not divisible by tp={world_size}")
        _run(
            world_size,
            _test_swap_matches_reference,
            pad_bs=pad_bs,
            n=n,
            vocab=vocab,
            dtype=dtype,
        )

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    @pytest.mark.parametrize("pad_bs,n,vocab", SHAPES)
    def test_swap_chain_safety(self, world_size, pad_bs, n, vocab):
        if pad_bs % world_size != 0:
            pytest.skip(f"pad_bs={pad_bs} not divisible by tp={world_size}")
        if vocab % world_size != 0:
            pytest.skip(f"vocab={vocab} not divisible by tp={world_size}")
        _run(
            world_size,
            _test_swap_chain_safety,
            pad_bs=pad_bs,
            n=n,
            vocab=vocab,
            dtype=torch.float32,
        )
