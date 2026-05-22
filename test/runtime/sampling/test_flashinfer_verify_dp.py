# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Multi-rank bit-equality test for the Batch-DP spec-verify path (M4):
``verify()`` with ``dp_sampling=True`` on a per-rank shard must return
the same ``(predict, accept_length)`` over real rows ``[0:bs]`` as the
full-batch path with ``dp_sampling=False`` on the gathered logits.
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
# Synthetic input generation
# ---------------------------------------------------------------------------


def _make_logits(bs: int, n: int, vocab: int, *, dtype, device, seed: int):
    # Same seed -> every rank materialises the identical tensor; DP path
    # then shards it, full-batch path uses it whole.
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.empty(bs * n, vocab, dtype=dtype, device=device).normal_(generator=g)


def _make_candidates(bs: int, n: int, vocab: int, *, device, seed: int):
    g = torch.Generator(device=device)
    g.manual_seed(seed + 1)
    return torch.randint(
        low=0, high=vocab, size=(bs, n), dtype=torch.int32, device=device, generator=g
    )


def _seed_pool_scalars(backend, *, bs: int, temperature: float, top_k: int, top_p: float):
    # Bypass _reset_slot / prepare_step; write pool rows directly so the
    # test stays independent of ScheduleBatch / SamplingParams construction.
    backend._temperature_pool[: bs + 1].fill_(temperature)
    backend._top_k_pool[: bs + 1].fill_(top_k)
    backend._top_p_pool[: bs + 1].fill_(top_p)


def _seed_coins(backend, *, bs: int, n: int, seed: int):
    # Identical coin buffers across ranks/paths -> deterministic chain
    # rejection sampling, so any output divergence is a real bug.
    g = torch.Generator(device=backend._coins_buf.device)
    g.manual_seed(seed + 7)
    backend._coins_buf[:bs, :n].uniform_(1e-6, 1.0, generator=g)
    backend._final_coins_buf[:bs].uniform_(1e-6, 1.0, generator=g)


# ---------------------------------------------------------------------------
# Worker bodies
# ---------------------------------------------------------------------------


def _build_backend(*, max_bs: int, max_n: int, vocab: int, device, group):
    from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
    from tokenspeed.runtime.sampling.backends.flashinfer import (
        FlashInferSamplingBackend,
    )

    cfg = SamplingBackendConfig(
        enable_nan_detection=False,
        enable_output_logprobs=False,
        max_bs=max_bs,
        max_draft_tokens_per_req=max_n,
        max_req_pool_size=max(max_bs, 4),
        vocab_size=vocab,
        device=device,
        random_seed=123,
        tp_group=group,
        # Off so the full-batch verify() does not broadcast: deterministic
        # kernels + identical seeds already give cross-rank bit-equality.
        enable_tp_sync=False,
    )
    return FlashInferSamplingBackend(cfg)


def _test_verify_dp_matches_today(
    rank,
    world_size,
    device,
    group,
    *,
    bs: int,
    n: int,
    vocab: int,
    is_all_greedy: bool,
    dtype,
):
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo

    tp_size = world_size
    pad_bs = ((bs + tp_size - 1) // tp_size) * tp_size
    k_req = pad_bs // tp_size

    backend = _build_backend(
        max_bs=max(bs, pad_bs),
        max_n=max(n, 1),
        vocab=vocab,
        device=device,
        group=group,
    )
    _seed_pool_scalars(backend, bs=bs, temperature=1.0, top_k=32, top_p=0.9)

    full_logits = _make_logits(bs, n, vocab, dtype=dtype, device=device, seed=2024)
    candidates = _make_candidates(bs, n, vocab, device=device, seed=2024)
    req_pool_indices = torch.arange(bs, dtype=torch.int64, device=device)

    sampling_info_full_batch = SamplingBatchInfo(
        is_all_greedy=is_all_greedy,
        vocab_size=vocab,
        req_pool_indices=req_pool_indices,
        device=str(device),
        dp_sampling=False,
    )
    sampling_info_dp = SamplingBatchInfo(
        is_all_greedy=is_all_greedy,
        vocab_size=vocab,
        req_pool_indices=req_pool_indices,
        device=str(device),
        dp_sampling=True,
    )

    class _StubOutput:
        pass

    _seed_coins(backend, bs=bs, n=n, seed=2024)
    full_batch_in = _StubOutput()
    # softmax+renorm mutate in-place; clone so the DP run sees pristine input.
    full_batch_in.next_token_logits = full_logits.clone()
    predict_full, accept_length_full = backend.verify(
        full_batch_in, sampling_info_full_batch, candidates
    )
    predict_full = predict_full.clone()
    accept_length_full = accept_length_full.clone()

    _seed_coins(backend, bs=bs, n=n, seed=2024)
    full_logits_padded = torch.nn.functional.pad(
        full_logits.view(bs, n, vocab), (0, 0, 0, 0, 0, pad_bs - bs)
    ).view(pad_bs * n, vocab)
    local_logits = full_logits_padded[
        rank * k_req * n : (rank + 1) * k_req * n
    ].clone()
    dp_in = _StubOutput()
    dp_in.next_token_logits = local_logits
    predict_dp, accept_length_dp = backend.verify(
        dp_in, sampling_info_dp, candidates
    )

    # Phantom rows ([bs:pad_bs]) are intentionally not checked.
    torch.testing.assert_close(predict_dp, predict_full, rtol=0, atol=0)
    torch.testing.assert_close(accept_length_dp, accept_length_full, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Public test cases
# ---------------------------------------------------------------------------


WORLD_SIZES = [
    pytest.param(2, id="tp2"),
    pytest.param(4, id="tp4"),
]

# (bs, N) -- bs covers both even (bs % tp == 0) and odd cases that
# exercise the phantom-padding path.
SHAPES = [
    pytest.param(8, 2, id="bs8_n2"),
    pytest.param(8, 4, id="bs8_n4"),
    pytest.param(9, 2, id="bs9_n2"),
    pytest.param(9, 4, id="bs9_n4"),
]


class TestFlashInferVerifyDP:
    """Bit-equality of ``verify()`` between the full-batch path and the
    DP path on the FlashInferSamplingBackend, across a world-size /
    bucket sweep.
    """

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    @pytest.mark.parametrize("bs,n", SHAPES)
    @pytest.mark.parametrize(
        "dtype",
        [pytest.param(torch.float32, id="fp32")],
    )
    def test_stochastic_path(self, world_size, bs, n, dtype):
        """Stochastic verify (chain_speculative_sampling_target_only)."""
        _run(
            world_size,
            _test_verify_dp_matches_today,
            bs=bs,
            n=n,
            vocab=256,
            is_all_greedy=False,
            dtype=dtype,
        )

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    @pytest.mark.parametrize("bs,n", SHAPES)
    def test_greedy_path(self, world_size, bs, n):
        """Greedy verify (verify_chain_greedy)."""
        _run(
            world_size,
            _test_verify_dp_matches_today,
            bs=bs,
            n=n,
            vocab=256,
            is_all_greedy=True,
            dtype=torch.float32,
        )
