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

"""Shape-swap helper for Batch-DP spec-sampling (NCCL fallback path).

Companion to the Batch-DP sampling pipeline laid out in
``bench/dp_sampling_flow.html`` and ``.skills/validate-collective-refactor/SKILL.md``.

Swaps which axis of the logits tensor is sharded:

  before, rank r:  [pad_bs * N, V/TP]   (full batch, my vocab shard)
  after,  rank r:  [K_req * N,  V]      (my K_req requests, all vocab)

The sharding unit on the bs axis is the **request** (size N rows), never the
flat row, because ``chain_speculative_sampling_target_only`` walks N causally
within a request. The caller is responsible for padding bs to ``pad_bs =
K_req * tp_size`` in whole-request units before calling this helper.

Cost per rank: one ``all_to_all_single`` ingress (TP-1 chunks of
``K_req * N * V/TP * dtype`` bytes) plus a ``permute + contiguous`` reorder
kernel.

This module exposes the bare NCCL implementation. Production callers should
prefer ``tokenspeed.runtime.distributed.dp_sampling_comm.DpSamplingComm``,
which owns a persistent workspace, can switch to a one-sided NVLink fast
path when available, and additionally provides stage-6 output gathering.
``DpSamplingComm`` delegates here for its NCCL fallback.
"""

from __future__ import annotations

import torch

from tokenspeed.runtime.distributed.comm_backend import CommBackend, Group
from tokenspeed.runtime.distributed.comm_ops import all_to_all_single


def swap_batch_vocab(
    local_logits: torch.Tensor,
    *,
    tp_size: int,
    pad_bs: int,
    num_tokens_per_req: int,
    vocab_size: int,
    rank: int,
    group: Group,
    backend: CommBackend | None = None,
) -> torch.Tensor:
    """[pad_bs * N, V/TP] (vocab-sharded, full batch)
       -> [K_req * N, V]  (batch-sharded, vocab-replicated).

    Args:
        local_logits: pre-padded local logits of shape
            ``(pad_bs * num_tokens_per_req, vocab_size // tp_size)``. Rows
            ``[bs * N : pad_bs * N]`` are phantom-request padding and may
            hold arbitrary values; they are routed to whichever rank ends
            up owning those phantom requests and are discarded downstream.
        tp_size: TP world size for ``group``.
        pad_bs: padded batch size, must be a multiple of ``tp_size``.
        num_tokens_per_req: N — the inner chain axis (1 for regular sample,
            ``spec_num_draft_tokens`` for spec verify).
        vocab_size: full vocab size V, must be a multiple of ``tp_size``.
        rank: caller's rank inside ``group``.
        group: TP comm group tuple.
        backend: comm backend override (defaults to the global backend).

    Returns:
        Tensor of shape ``(K_req * num_tokens_per_req, vocab_size)`` where
        ``K_req = pad_bs // tp_size``. Row ``r_local * N + d`` of the
        return corresponds to global request ``rank * K_req + r_local``,
        draft position ``d``.
    """
    assert (
        pad_bs % tp_size == 0
    ), f"swap_batch_vocab: pad_bs={pad_bs} must be divisible by tp_size={tp_size}"
    assert (
        vocab_size % tp_size == 0
    ), f"swap_batch_vocab: vocab_size={vocab_size} must be divisible by tp_size={tp_size}"

    k_req = pad_bs // tp_size
    v_local = vocab_size // tp_size
    n = num_tokens_per_req

    expected_shape = (pad_bs * n, v_local)
    assert tuple(local_logits.shape) == expected_shape, (
        f"swap_batch_vocab: local_logits shape {tuple(local_logits.shape)} "
        f"!= expected {expected_shape} (pad_bs={pad_bs}, N={n}, V/TP={v_local})"
    )

    # Even-split a2a along dim 0: chunk r is the slice
    # local_logits[r*K_req*N : (r+1)*K_req*N], which is exactly the K_req
    # requests (each contiguous N rows) destined for rank r. The N draft
    # positions of every request stay inside one chunk -> chain causality
    # preserved.
    recv = torch.empty_like(local_logits)
    all_to_all_single(recv, local_logits, rank, group, backend=backend)

    # After a2a: recv[src*K_req*N : (src+1)*K_req*N] = source rank `src`'s
    # vocab shard for my K_req requests. The leading dim therefore indexes
    # source-rank == vocab-shard. Permute brings that next to the V/TP cols,
    # and the final view concatenates them into a single V axis.
    return (
        recv.view(tp_size, k_req, n, v_local)
        .permute(1, 2, 0, 3)
        .contiguous()
        .view(k_req * n, vocab_size)
    )
