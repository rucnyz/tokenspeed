"""Tests for tokenspeed.runtime.distributed.dp_sampling_comm.

Validates ``DpSamplingComm`` end-to-end across real multi-rank NCCL workers.
Three properties are tested per worker:

  1. Stage-4 swap matches the standalone ``swap_batch_vocab`` helper
     bit-for-bit (parity between the class API and the existing free
     function -- so production callers can migrate without behavior
     change).

  2. Stage-6 ``gather_verify_outputs`` returns the global concatenation
     of per-rank inputs in source-rank order, exactly as 3 separate
     ``all_gather_into_tensor`` calls would.

  3. Full swap + sample-stub + gather cycle captures cleanly into
     ``torch.cuda.graph()`` and is bit-identical on replay -- the
     decode-loop graph-safety guarantee.

The fast-path (``backend="onesided"``) is exercised when the local
PyTorch/platform combo supports symmetric memory; otherwise those cases
skip cleanly and NCCL remains the reference path.

Usage:
    python -m pytest test/runtime/distributed/test_dp_sampling_comm.py -v
"""

import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


# ---------------------------------------------------------------------------
# Multi-rank harness (mirrors test_dp_sampling_swap.py)
# ---------------------------------------------------------------------------


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
# Fixtures shared across tests
# ---------------------------------------------------------------------------


def _onesided_available_for_test(group) -> bool:
    try:
        from tokenspeed.runtime.distributed.dp_sampling_comm import _onesided_available

        return _onesided_available(group)
    except Exception:
        return False


def _build_comm(rank, world_size, group, *, pad_bs, n, vocab, dtype, backend):
    from tokenspeed.runtime.distributed.dp_sampling_comm import DpSamplingComm

    return DpSamplingComm(
        tp_size=world_size,
        rank=rank,
        group=group,
        max_pad_bs=pad_bs,
        num_tokens_per_req=n,
        vocab_size=vocab,
        logits_dtype=dtype,
        backend=backend,
    )


def _ground_truth_full_logits(pad_bs, n, vocab, *, dtype, device):
    """[pad_bs * N, V] arange tensor; each cell is globally-unique."""
    return torch.arange(pad_bs * n * vocab, dtype=dtype, device=device).view(
        pad_bs * n, vocab
    )


# ---------------------------------------------------------------------------
# Worker test bodies
# ---------------------------------------------------------------------------


def _test_swap_parity_with_free_function(
    rank, world_size, device, group, *, pad_bs, n, vocab, dtype, backend
):
    """Stage 4 via DpSamplingComm == stage 4 via the standalone helper."""
    from tokenspeed.runtime.distributed.dp_sampling_swap import swap_batch_vocab

    tp = world_size
    v_local = vocab // tp

    full = _ground_truth_full_logits(pad_bs, n, vocab, dtype=dtype, device=device)
    local_logits = full[:, rank * v_local : (rank + 1) * v_local].contiguous()

    comm = _build_comm(
        rank,
        world_size,
        group,
        pad_bs=pad_bs,
        n=n,
        vocab=vocab,
        dtype=dtype,
        backend=backend,
    )
    assert comm.backend == backend
    assert comm.fast_path_enabled is (backend == "onesided")

    out_class = comm.swap_batch_vocab(local_logits, pad_bs=pad_bs)

    out_free = swap_batch_vocab(
        local_logits,
        tp_size=tp,
        pad_bs=pad_bs,
        num_tokens_per_req=n,
        vocab_size=vocab,
        rank=rank,
        group=group,
    )

    assert out_class.shape == out_free.shape
    torch.testing.assert_close(out_class, out_free)


def _test_gather_verify_outputs_correctness(
    rank, world_size, device, group, *, pad_bs, n, backend
):
    """Stage 6: 3 per-rank tensors -> contiguous full-batch concatenation."""
    tp = world_size
    k_req = pad_bs // tp

    comm = _build_comm(
        rank,
        world_size,
        group,
        pad_bs=pad_bs,
        n=n,
        vocab=tp * 4,
        dtype=torch.bfloat16,
        backend=backend,
    )

    predict_local = torch.arange(
        rank * k_req * n, (rank + 1) * k_req * n, dtype=torch.int32, device=device
    ).view(k_req, n)
    accept_index_local = (predict_local * 2 + 1).contiguous()
    accept_length_local = torch.arange(
        rank * k_req, (rank + 1) * k_req, dtype=torch.int32, device=device
    )

    predict_full, accept_index_full, accept_length_full = comm.gather_verify_outputs(
        predict_local,
        accept_index_local,
        accept_length_local,
        pad_bs=pad_bs,
    )

    expected_predict = torch.arange(
        0, pad_bs * n, dtype=torch.int32, device=device
    ).view(pad_bs, n)
    expected_accept_index = expected_predict * 2 + 1
    expected_accept_length = torch.arange(0, pad_bs, dtype=torch.int32, device=device)

    torch.testing.assert_close(predict_full, expected_predict)
    torch.testing.assert_close(accept_index_full, expected_accept_index)
    torch.testing.assert_close(accept_length_full, expected_accept_length)


def _test_gather_persistent_buffer_reuse(
    rank, world_size, device, group, *, pad_bs, n, backend
):
    """Stage 6 returns aliases into persistent storage -- repeated calls
    must reuse the same underlying storage so the buffer addresses stay
    stable across captured-graph replays.
    """
    tp = world_size
    k_req = pad_bs // tp

    comm = _build_comm(
        rank,
        world_size,
        group,
        pad_bs=pad_bs,
        n=n,
        vocab=tp * 4,
        dtype=torch.bfloat16,
        backend=backend,
    )

    predict_local = torch.zeros(k_req, n, dtype=torch.int32, device=device)
    accept_index_local = torch.zeros(k_req, n, dtype=torch.int32, device=device)
    accept_length_local = torch.zeros(k_req, dtype=torch.int32, device=device)

    p1, ai1, al1 = comm.gather_verify_outputs(
        predict_local, accept_index_local, accept_length_local, pad_bs=pad_bs
    )
    p2, ai2, al2 = comm.gather_verify_outputs(
        predict_local, accept_index_local, accept_length_local, pad_bs=pad_bs
    )

    # ``data_ptr()`` equality proves both outputs alias the same persistent
    # tensor -- the property a CUDA graph relies on across replays.
    assert p1.data_ptr() == p2.data_ptr()
    assert ai1.data_ptr() == ai2.data_ptr()
    assert al1.data_ptr() == al2.data_ptr()


def _test_swap_and_gather_cuda_graph_replay(
    rank, world_size, device, group, *, pad_bs, n, vocab, backend
):
    """The graph-safety contract: capture one (swap + sample-stub +
    gather) pass into ``torch.cuda.graph()`` then replay N times with
    different input values. Every replay must produce bit-identical
    output to a non-graphed reference run.
    """
    tp = world_size
    k_req = pad_bs // tp
    v_local = vocab // tp

    comm = _build_comm(
        rank,
        world_size,
        group,
        pad_bs=pad_bs,
        n=n,
        vocab=vocab,
        dtype=torch.float32,
        backend=backend,
    )

    # Persistent input buffers -- the addresses captured by the graph.
    local_logits_buf = torch.empty(
        pad_bs * n, v_local, dtype=torch.float32, device=device
    )
    predict_local_buf = torch.empty(k_req, n, dtype=torch.int32, device=device)
    accept_index_local_buf = torch.empty(k_req, n, dtype=torch.int32, device=device)
    accept_length_local_buf = torch.empty(k_req, dtype=torch.int32, device=device)

    def _fill_inputs(step: int):
        full = _ground_truth_full_logits(
            pad_bs, n, vocab, dtype=torch.float32, device=device
        )
        full = full + step * 1000.0
        local_logits_buf.copy_(
            full[:, rank * v_local : (rank + 1) * v_local].contiguous()
        )
        # The "sampled" outputs are derived deterministically from the step
        # so we can verify graph replay didn't capture stale values.
        predict_local_buf.copy_(
            torch.arange(
                rank * k_req * n, (rank + 1) * k_req * n,
                dtype=torch.int32, device=device,
            ).view(k_req, n) + step
        )
        accept_index_local_buf.copy_(predict_local_buf * 2)
        accept_length_local_buf.copy_(
            torch.arange(
                rank * k_req, (rank + 1) * k_req,
                dtype=torch.int32, device=device,
            ) + step
        )

    def _run_one_step():
        swapped = comm.swap_batch_vocab(local_logits_buf, pad_bs=pad_bs)
        p, ai, al = comm.gather_verify_outputs(
            predict_local_buf,
            accept_index_local_buf,
            accept_length_local_buf,
            pad_bs=pad_bs,
        )
        return swapped, p, ai, al

    # Warm NCCL on a side stream so allocator state is stable, then capture.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _fill_inputs(step=0)
        for _ in range(3):
            _run_one_step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize(device)
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    _fill_inputs(step=0)
    with torch.cuda.graph(graph, stream=side):
        swapped_captured, p_captured, ai_captured, al_captured = _run_one_step()
    torch.cuda.synchronize(device)
    dist.barrier()

    # Replay 5 times with different input values; compare against a fresh
    # non-graphed run with the same inputs.
    for step in range(1, 6):
        _fill_inputs(step=step)
        graph.replay()
        torch.cuda.synchronize(device)

        graph_swapped = swapped_captured.clone()
        graph_predict = p_captured.clone()
        graph_accept_index = ai_captured.clone()
        graph_accept_length = al_captured.clone()
        dist.barrier()

        _fill_inputs(step=step)
        ref_swapped, ref_predict, ref_accept_index, ref_accept_length = _run_one_step()
        torch.cuda.synchronize(device)

        torch.testing.assert_close(graph_swapped, ref_swapped)
        torch.testing.assert_close(graph_predict, ref_predict)
        torch.testing.assert_close(graph_accept_index, ref_accept_index)
        torch.testing.assert_close(graph_accept_length, ref_accept_length)
        dist.barrier()


def _test_onesided_matches_nccl(
    rank, world_size, device, group, *, pad_bs, n, vocab, dtype
):
    tp = world_size
    v_local = vocab // tp
    k_req = pad_bs // tp

    full = _ground_truth_full_logits(pad_bs, n, vocab, dtype=dtype, device=device)
    local_logits = full[:, rank * v_local : (rank + 1) * v_local].contiguous()

    nccl_comm = _build_comm(
        rank,
        world_size,
        group,
        pad_bs=pad_bs,
        n=n,
        vocab=vocab,
        dtype=dtype,
        backend="nccl",
    )
    onesided_comm = _build_comm(
        rank,
        world_size,
        group,
        pad_bs=pad_bs,
        n=n,
        vocab=vocab,
        dtype=dtype,
        backend="onesided",
    )

    torch.testing.assert_close(
        onesided_comm.swap_batch_vocab(local_logits, pad_bs=pad_bs),
        nccl_comm.swap_batch_vocab(local_logits, pad_bs=pad_bs),
    )

    predict_local = torch.arange(
        rank * k_req * n, (rank + 1) * k_req * n, dtype=torch.int32, device=device
    ).view(k_req, n)
    accept_index_local = (predict_local * 3 + 7).contiguous()
    accept_length_local = torch.arange(
        rank * k_req, (rank + 1) * k_req, dtype=torch.int32, device=device
    )

    onesided_outputs = onesided_comm.gather_verify_outputs(
        predict_local, accept_index_local, accept_length_local, pad_bs=pad_bs
    )
    nccl_outputs = nccl_comm.gather_verify_outputs(
        predict_local, accept_index_local, accept_length_local, pad_bs=pad_bs
    )
    for actual, expected in zip(onesided_outputs, nccl_outputs, strict=True):
        torch.testing.assert_close(actual, expected)


# ---------------------------------------------------------------------------
# Public test cases
# ---------------------------------------------------------------------------

WORLD_SIZES = [
    pytest.param(2, id="tp2"),
    pytest.param(4, id="tp4"),
]

SHAPES = [
    pytest.param(8, 1, 64, id="sample_pad_bs8"),
    pytest.param(8, 4, 64, id="spec_pad_bs8_n4"),
    pytest.param(12, 3, 96, id="spec_pad_bs12_n3"),
]

DTYPES = [
    pytest.param(torch.float32, id="fp32"),
    pytest.param(torch.bfloat16, id="bf16"),
]

BACKENDS = [
    pytest.param("nccl", id="nccl"),
    pytest.param("onesided", id="onesided"),
]


class TestDpSamplingComm:

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    @pytest.mark.parametrize("pad_bs,n,vocab", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_swap_parity_with_free_function(
        self, world_size, pad_bs, n, vocab, dtype, backend
    ):
        if pad_bs % world_size != 0 or vocab % world_size != 0:
            pytest.skip("shape not divisible by tp")
        if backend == "onesided" and not _onesided_available_for_test(
            tuple(range(world_size))
        ):
            pytest.skip("one-sided dp-sampling backend is not available")
        _run(
            world_size,
            _test_swap_parity_with_free_function,
            pad_bs=pad_bs,
            n=n,
            vocab=vocab,
            dtype=dtype,
            backend=backend,
        )

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    @pytest.mark.parametrize("pad_bs,n", [(8, 1), (8, 4), (12, 3)])
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_gather_verify_outputs_correctness(self, world_size, pad_bs, n, backend):
        if pad_bs % world_size != 0:
            pytest.skip("pad_bs not divisible by tp")
        if backend == "onesided" and not _onesided_available_for_test(
            tuple(range(world_size))
        ):
            pytest.skip("one-sided dp-sampling backend is not available")
        _run(
            world_size,
            _test_gather_verify_outputs_correctness,
            pad_bs=pad_bs,
            n=n,
            backend=backend,
        )

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_gather_persistent_buffer_reuse(self, world_size, backend):
        if backend == "onesided" and not _onesided_available_for_test(
            tuple(range(world_size))
        ):
            pytest.skip("one-sided dp-sampling backend is not available")
        _run(
            world_size,
            _test_gather_persistent_buffer_reuse,
            pad_bs=8,
            n=2,
            backend=backend,
        )

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_swap_and_gather_cuda_graph_replay(self, world_size, backend):
        if backend == "onesided" and not _onesided_available_for_test(
            tuple(range(world_size))
        ):
            pytest.skip("one-sided dp-sampling backend is not available")
        _run(
            world_size,
            _test_swap_and_gather_cuda_graph_replay,
            pad_bs=8,
            n=2,
            vocab=64,
            backend=backend,
        )

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    @pytest.mark.parametrize("pad_bs,n,vocab", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_onesided_matches_nccl(self, world_size, pad_bs, n, vocab, dtype):
        if pad_bs % world_size != 0 or vocab % world_size != 0:
            pytest.skip("shape not divisible by tp")
        if not _onesided_available_for_test(tuple(range(world_size))):
            pytest.skip("one-sided dp-sampling backend is not available")
        _run(
            world_size,
            _test_onesided_matches_nccl,
            pad_bs=pad_bs,
            n=n,
            vocab=vocab,
            dtype=dtype,
        )
