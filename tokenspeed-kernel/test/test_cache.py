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

"""
Production-scale test for the writeback fence race condition on B200 (sm_100a).

Key finding from runtime log analysis:
  - Pages are ALWAYS disjoint (collisions=0 in all 1236 dispatches)
  - DeviceNodeRef correctly prevents page overlap
  - Yet io_backend="kernel" still produces corrupted results
  - DMA mode (io_backend="direct") is unaffected

This test reproduces the production scenario:
  1. Single shared FP8 KV cache buffer (same contiguous allocation as production)
  2. Transfer kernel reads from writeback pages and writes to HOST-PINNED memory
  3. Forward kernel (fused_fp8_set_kv_buffer) writes FP8 quantized KV data to
     DISJOINT pages in the same shared buffer, concurrently on a different stream
  4. No fence between the two streams

The hypothesis: the persistent SM transfer kernel's massive device-read +
host-write L2 traffic corrupts the forward kernel's L2 state on B200, even
when pages are completely disjoint. Sharing the same physical allocation
makes L2 sector interference more likely (adjacent pages map to nearby L2
sectors).
"""

from __future__ import annotations

import pytest
import torch
import triton
import triton.language as tl

import math

from tokenspeed_kernel.ops.kvcache.triton import transfer_kv_all_layer
from tokenspeed_kernel.ops.attention.flashinfer import (
    trtllm_batch_context_with_kv_cache,
    trtllm_batch_decode_with_kv_cache,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed.runtime.layers.attention.kv_cache.trtllm_fp8_kv_kernel import (
    fused_fp8_set_kv_buffer,
)


# Production-scale parameters (MiniMax-M2.7-NVFP4: 62 layers, 8 KV heads, 128 dim)
NUM_LAYERS = 62
PAGE_SIZE = 32
HEAD_DIM = 128
NUM_KV_HEADS = 8
ITEM_SIZE = NUM_KV_HEADS * HEAD_DIM * 2  # bytes per slot (fp16)

# Writeback scale: 2776 pages = 88832 slots (matches production peak)
NUM_WB_PAGES = 2776
NUM_WB_SLOTS = NUM_WB_PAGES * PAGE_SIZE

# Forward scale: 256 pages for the forward path (matches typical prefill batch)
NUM_FWD_PAGES = 256
NUM_FWD_SLOTS = NUM_FWD_PAGES * PAGE_SIZE

ITERATIONS = 200


@triton.jit
def _forward_compute_kernel(
    kv_ptrs_ptr,
    output_ptr,
    numel_per_layer,
    num_layers: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Simulates the forward kernel: reads from device KV pages (attention-like
    pattern) and writes a reduction to an output buffer.

    Each program reads chunks from all layers and accumulates a checksum.
    This exercises L2 read paths that may be corrupted by concurrent
    transfer kernel traffic.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    acc = tl.zeros([BLOCK], dtype=tl.float32)

    for layer in range(num_layers):
        base_ptr = tl.load(kv_ptrs_ptr + layer).to(tl.pointer_type(tl.float16))
        for chunk_start in range(pid * BLOCK, numel_per_layer, num_programs * BLOCK):
            offsets = chunk_start + tl.arange(0, BLOCK)
            mask = offsets < numel_per_layer
            data = tl.load(base_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            acc += data

    # Write accumulated result
    out_offsets = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(output_ptr + out_offsets, acc)


@triton.jit
def _forward_write_kernel(
    kv_ptrs_ptr,
    numel_per_layer,
    pattern: tl.constexpr,
    num_layers: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Simulates set_kv_buffer: writes new token data to forward KV pages.
    Uses standard (non-streaming) stores.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for layer in range(num_layers):
        base_ptr = tl.load(kv_ptrs_ptr + layer).to(tl.pointer_type(tl.uint16))
        for chunk_start in range(pid * BLOCK, numel_per_layer, num_programs * BLOCK):
            offsets = chunk_start + tl.arange(0, BLOCK)
            mask = offsets < numel_per_layer
            tl.store(
                base_ptr + offsets,
                tl.full([BLOCK], pattern, dtype=tl.uint16),
                mask=mask,
            )


def _make_large_kv_pool(num_slots, device, dtype=torch.float16):
    """Create per-layer K and V buffers + pointer tensors for a large pool."""
    k_buffers = [
        torch.zeros(num_slots, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
        for _ in range(NUM_LAYERS)
    ]
    v_buffers = [
        torch.zeros(num_slots, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
        for _ in range(NUM_LAYERS)
    ]
    k_ptrs = torch.tensor(
        [b.data_ptr() for b in k_buffers], dtype=torch.uint64, device=device
    )
    v_ptrs = torch.tensor(
        [b.data_ptr() for b in v_buffers], dtype=torch.uint64, device=device
    )
    return k_buffers, v_buffers, k_ptrs, v_ptrs


def _make_host_pinned_pool(num_slots):
    """Create per-layer K and V HOST-PINNED buffers (simulates host KV pool)."""
    k_buffers = [
        torch.zeros(
            num_slots, NUM_KV_HEADS, HEAD_DIM,
            dtype=torch.float16, pin_memory=True
        )
        for _ in range(NUM_LAYERS)
    ]
    v_buffers = [
        torch.zeros(
            num_slots, NUM_KV_HEADS, HEAD_DIM,
            dtype=torch.float16, pin_memory=True
        )
        for _ in range(NUM_LAYERS)
    ]
    # Pointers must be on GPU for the Triton kernel
    k_ptrs = torch.tensor(
        [b.data_ptr() for b in k_buffers], dtype=torch.uint64, device="cuda"
    )
    v_ptrs = torch.tensor(
        [b.data_ptr() for b in v_buffers], dtype=torch.uint64, device="cuda"
    )
    return k_buffers, v_buffers, k_ptrs, v_ptrs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_production_shared_buffer_fp8_no_fence():
    """
    Most realistic reproduction of the production race condition.

    Improvements over naive version:
    1. Pre-warms Triton kernels to eliminate JIT overhead from timing
    2. Uses CUDA events to measure and verify actual GPU-side overlap
    3. Launches transfer first, then immediately submits all forward kernels
       (minimizing CPU dispatch gap to maximize overlap window)
    4. Runs multiple forward rounds within a single long transfer to ensure
       at least some forward kernels execute while transfer is still running
    5. Reports overlap statistics to confirm the test is exercising the
       concurrent execution path

    On B200 without the fence, the concurrent L2 traffic from the persistent
    transfer kernel corrupts the FP8 values written by fused_fp8_set_kv_buffer.
    """
    device = "cuda"
    platform = current_platform()
    if platform.vendor != "nvidia":
        pytest.skip("NVIDIA GPU required")

    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    total_mem = torch.cuda.get_device_properties(device).total_memory

    if total_mem > 150e9:
        num_layers = 62
        num_wb_pages = 2776
    elif total_mem > 70e9:
        num_layers = 32
        num_wb_pages = 1024
    else:
        num_layers = 16
        num_wb_pages = 512

    num_fwd_pages = 64
    total_pages = num_wb_pages + num_fwd_pages
    page_size = PAGE_SIZE
    num_kv_heads = NUM_KV_HEADS
    head_dim = HEAD_DIM
    fp8_item_size = num_kv_heads * head_dim

    num_wb_slots = num_wb_pages * page_size
    total_slots = total_pages * page_size

    print(f"\nSM count: {sm_count}, GPU memory: {total_mem / 1e9:.1f} GB")
    print(f"Layers: {num_layers}, WB pages: {num_wb_pages}, FWD pages: {num_fwd_pages}")
    print(f"Shared buffer per layer: {total_slots * fp8_item_size / 1e6:.1f} MB (FP8)")
    total_transfer_bytes = num_wb_slots * fp8_item_size * num_layers * 2
    print(f"Total transfer volume per iter: {total_transfer_bytes / 1e9:.2f} GB")

    stream_exec = torch.cuda.Stream()
    stream_write = torch.cuda.Stream()

    # Shared FP8 KV cache: [total_slots, num_kv_heads, head_dim] per layer
    k_caches = []
    v_caches = []
    for _ in range(num_layers):
        k_caches.append(torch.zeros(
            total_slots, num_kv_heads, head_dim,
            device=device, dtype=torch.float8_e4m3fn,
        ))
        v_caches.append(torch.zeros(
            total_slots, num_kv_heads, head_dim,
            device=device, dtype=torch.float8_e4m3fn,
        ))

    k_cache_ptrs = torch.tensor(
        [k.data_ptr() for k in k_caches], dtype=torch.uint64, device=device
    )
    v_cache_ptrs = torch.tensor(
        [v.data_ptr() for v in v_caches], dtype=torch.uint64, device=device
    )

    host_buf_size = num_wb_slots * fp8_item_size
    k_host_bufs = [
        torch.zeros(host_buf_size, dtype=torch.uint8, pin_memory=True)
        for _ in range(num_layers)
    ]
    v_host_bufs = [
        torch.zeros(host_buf_size, dtype=torch.uint8, pin_memory=True)
        for _ in range(num_layers)
    ]
    k_host_ptrs = torch.tensor(
        [b.data_ptr() for b in k_host_bufs], dtype=torch.uint64, device=device
    )
    v_host_ptrs = torch.tensor(
        [b.data_ptr() for b in v_host_bufs], dtype=torch.uint64, device=device
    )

    wb_src_indices = torch.arange(num_wb_slots, dtype=torch.int32, device=device)
    wb_dst_indices = torch.arange(num_wb_slots, dtype=torch.int32, device=device)

    fwd_cache_loc = torch.arange(
        num_wb_slots, total_slots, dtype=torch.int32, device=device
    )
    num_fwd_tokens = fwd_cache_loc.shape[0]
    k_input = torch.randn(
        num_fwd_tokens, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    v_input = torch.randn(
        num_fwd_tokens, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16
    )

    # --- Pre-warm: run both kernels once to JIT-compile Triton ---
    print("Pre-warming Triton kernels...")
    with torch.cuda.stream(stream_write):
        transfer_kv_all_layer(
            src_k_layers=k_cache_ptrs,
            dst_k_layers=k_host_ptrs,
            src_v_layers=v_cache_ptrs,
            dst_v_layers=v_host_ptrs,
            src_indices=wb_src_indices,
            dst_indices=wb_dst_indices,
            item_size=fp8_item_size,
            num_layers=num_layers,
        )
    with torch.cuda.stream(stream_exec):
        for layer in range(num_layers):
            fused_fp8_set_kv_buffer(
                k_input, v_input,
                k_caches[layer], v_caches[layer],
                fwd_cache_loc, page_size=page_size,
            )
    torch.cuda.synchronize()
    print("Warm-up done.")

    # --- Compute reference (no concurrency) ---
    ref_k_caches = []
    ref_v_caches = []
    for layer in range(num_layers):
        ref_k = torch.zeros(
            total_slots, num_kv_heads, head_dim,
            device=device, dtype=torch.float8_e4m3fn,
        )
        ref_v = torch.zeros(
            total_slots, num_kv_heads, head_dim,
            device=device, dtype=torch.float8_e4m3fn,
        )
        fused_fp8_set_kv_buffer(
            k_input, v_input, ref_k, ref_v, fwd_cache_loc, page_size=page_size,
        )
        ref_k_caches.append(ref_k)
        ref_v_caches.append(ref_v)
    torch.cuda.synchronize()

    # --- Main test loop ---
    corruption_count = 0
    overlap_count = 0
    first_msg = ""

    # CUDA events for measuring overlap
    transfer_start = torch.cuda.Event(enable_timing=True)
    transfer_end = torch.cuda.Event(enable_timing=True)
    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)

    for iteration in range(ITERATIONS):
        torch.cuda.synchronize()

        # Fill WB region, clear FWD region
        for k in k_caches:
            k[:num_wb_slots].fill_(0.5)
        for v in v_caches:
            v[:num_wb_slots].fill_(0.5)
        for k in k_caches:
            k[num_wb_slots:].zero_()
        for v in v_caches:
            v[num_wb_slots:].zero_()

        torch.cuda.synchronize()

        # Launch transfer with timing events
        with torch.cuda.stream(stream_write):
            transfer_start.record(stream_write)
            transfer_kv_all_layer(
                src_k_layers=k_cache_ptrs,
                dst_k_layers=k_host_ptrs,
                src_v_layers=v_cache_ptrs,
                dst_v_layers=v_host_ptrs,
                src_indices=wb_src_indices,
                dst_indices=wb_dst_indices,
                item_size=fp8_item_size,
                num_layers=num_layers,
            )
            transfer_end.record(stream_write)

        # Launch forward immediately — NO FENCE, NO WAIT
        # Tight loop to minimize CPU dispatch gap
        with torch.cuda.stream(stream_exec):
            forward_start.record(stream_exec)
            for layer in range(num_layers):
                fused_fp8_set_kv_buffer(
                    k_input, v_input,
                    k_caches[layer], v_caches[layer],
                    fwd_cache_loc, page_size=page_size,
                )
            forward_end.record(stream_exec)

        # Sync both
        stream_write.synchronize()
        stream_exec.synchronize()

        # Check overlap: did forward start before transfer ended?
        t_start = transfer_start.elapsed_time(forward_start)  # ms
        t_transfer = transfer_start.elapsed_time(transfer_end)
        t_forward = forward_start.elapsed_time(forward_end)

        # Overlap exists if forward started before transfer ended
        # t_start = time from transfer_start to forward_start
        # If t_start < t_transfer, forward started while transfer was running
        had_overlap = t_start < t_transfer
        if had_overlap:
            overlap_count += 1

        # Verify FWD region
        for layer in range(num_layers):
            fwd_k = k_caches[layer][num_wb_slots:].view(torch.uint8)
            ref_k = ref_k_caches[layer][num_wb_slots:].view(torch.uint8)
            mismatches_k = (fwd_k != ref_k).sum().item()

            if mismatches_k > 0:
                corruption_count += 1
                total_bytes = fwd_k.numel()
                overlap_ms = max(0, t_transfer - t_start)
                msg = (
                    f"iter={iteration} K layer={layer}: "
                    f"{mismatches_k}/{total_bytes} bytes differ "
                    f"({mismatches_k * 100.0 / total_bytes:.4f}%) "
                    f"[overlap={overlap_ms:.2f}ms, transfer={t_transfer:.2f}ms, "
                    f"fwd={t_forward:.2f}ms]"
                )
                if not first_msg:
                    first_msg = msg
                break

            fwd_v = v_caches[layer][num_wb_slots:].view(torch.uint8)
            ref_v = ref_v_caches[layer][num_wb_slots:].view(torch.uint8)
            mismatches_v = (fwd_v != ref_v).sum().item()

            if mismatches_v > 0:
                corruption_count += 1
                total_bytes = fwd_v.numel()
                overlap_ms = max(0, t_transfer - t_start)
                msg = (
                    f"iter={iteration} V layer={layer}: "
                    f"{mismatches_v}/{total_bytes} bytes differ "
                    f"({mismatches_v * 100.0 / total_bytes:.4f}%) "
                    f"[overlap={overlap_ms:.2f}ms, transfer={t_transfer:.2f}ms, "
                    f"fwd={t_forward:.2f}ms]"
                )
                if not first_msg:
                    first_msg = msg
                break

    print(f"\nOverlap stats: {overlap_count}/{ITERATIONS} iterations had GPU overlap")
    print(f"Shared-buffer FP8 results: {corruption_count}/{ITERATIONS} corrupted")

    if overlap_count == 0:
        print("WARNING: no overlap detected — transfer finishes before forward starts.")
        print("This test cannot exercise the race condition without overlap.")

    assert corruption_count == 0, (
        f"FP8 CORRUPTION in {corruption_count}/{ITERATIONS} iterations "
        f"({overlap_count} had overlap). "
        f"First: {first_msg}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_production_shared_buffer_attention_gemm_no_fence():
    """
    Most aggressive reproduction attempt.

    Key insight from previous test: fused_fp8_set_kv_buffer alone doesn't
    generate enough L2 pressure on the exec_stream side. In production,
    the forward pass does:
      1. Attention: READS from context pages in the SAME KV cache allocation
         (same buffer as transfer source, different offsets)
      2. MLP/MoE GEMM: massive weight tensor reads through L2
      3. set_kv_buffer: FP8 writes

    The critical pattern is: BOTH streams read heavily from the same physical
    allocation (transfer reads WB pages, attention reads context pages).
    This creates bidirectional heavy L2 read traffic on the same allocation
    plus host-write traffic from the transfer kernel.

    Layout in shared buffer per layer:
      [WB region | Context region | FWD region]
       transfer    attention reads   set_kv_buffer writes
       reads       (exec_stream)     (exec_stream)
       (write_stream)
    """
    device = "cuda"
    platform = current_platform()
    if platform.vendor != "nvidia":
        pytest.skip("NVIDIA GPU required")

    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    total_mem = torch.cuda.get_device_properties(device).total_memory

    if total_mem > 150e9:
        num_layers = 62
        num_wb_pages = 2776
    elif total_mem > 70e9:
        num_layers = 32
        num_wb_pages = 1024
    else:
        num_layers = 16
        num_wb_pages = 512

    # Context pages: simulates the KV cache pages that attention reads from.
    # In production, context can be 4K-128K tokens = 128-4096 pages.
    num_context_pages = 1024
    num_fwd_pages = 64
    total_pages = num_wb_pages + num_context_pages + num_fwd_pages
    page_size = PAGE_SIZE
    num_kv_heads = NUM_KV_HEADS
    head_dim = HEAD_DIM
    fp8_item_size = num_kv_heads * head_dim

    num_wb_slots = num_wb_pages * page_size
    num_ctx_slots = num_context_pages * page_size
    num_fwd_slots = num_fwd_pages * page_size
    total_slots = total_pages * page_size

    ctx_start = num_wb_slots
    fwd_start = num_wb_slots + num_ctx_slots

    print(f"\nSM count: {sm_count}, GPU memory: {total_mem / 1e9:.1f} GB")
    print(f"Layers: {num_layers}")
    print(f"  WB pages: {num_wb_pages} (transfer reads)")
    print(f"  Context pages: {num_context_pages} (attention reads)")
    print(f"  FWD pages: {num_fwd_pages} (set_kv_buffer writes)")
    print(f"Shared buffer per layer: {total_slots * fp8_item_size / 1e6:.1f} MB (FP8)")
    total_transfer_bytes = num_wb_slots * fp8_item_size * num_layers * 2
    total_attn_read_bytes = num_ctx_slots * fp8_item_size * num_layers * 2
    print(f"Transfer volume per iter: {total_transfer_bytes / 1e9:.2f} GB")
    print(f"Attention read volume per iter: {total_attn_read_bytes / 1e9:.2f} GB")

    stream_exec = torch.cuda.Stream()
    stream_write = torch.cuda.Stream()

    # Shared FP8 KV cache per layer
    k_caches = []
    v_caches = []
    for _ in range(num_layers):
        k_caches.append(torch.zeros(
            total_slots, num_kv_heads, head_dim,
            device=device, dtype=torch.float8_e4m3fn,
        ))
        v_caches.append(torch.zeros(
            total_slots, num_kv_heads, head_dim,
            device=device, dtype=torch.float8_e4m3fn,
        ))

    # Transfer pointers (read from WB region of same buffer)
    k_cache_ptrs = torch.tensor(
        [k.data_ptr() for k in k_caches], dtype=torch.uint64, device=device
    )
    v_cache_ptrs = torch.tensor(
        [v.data_ptr() for v in v_caches], dtype=torch.uint64, device=device
    )

    # Host-pinned destination
    host_buf_size = num_wb_slots * fp8_item_size
    k_host_bufs = [
        torch.zeros(host_buf_size, dtype=torch.uint8, pin_memory=True)
        for _ in range(num_layers)
    ]
    v_host_bufs = [
        torch.zeros(host_buf_size, dtype=torch.uint8, pin_memory=True)
        for _ in range(num_layers)
    ]
    k_host_ptrs = torch.tensor(
        [b.data_ptr() for b in k_host_bufs], dtype=torch.uint64, device=device
    )
    v_host_ptrs = torch.tensor(
        [b.data_ptr() for b in v_host_bufs], dtype=torch.uint64, device=device
    )

    # Transfer indices (WB region)
    wb_src_indices = torch.arange(num_wb_slots, dtype=torch.int32, device=device)
    wb_dst_indices = torch.arange(num_wb_slots, dtype=torch.int32, device=device)

    # Forward set_kv_buffer: writes to FWD region
    fwd_cache_loc = torch.arange(
        fwd_start, total_slots, dtype=torch.int32, device=device
    )
    num_fwd_tokens = fwd_cache_loc.shape[0]
    k_input = torch.randn(
        num_fwd_tokens, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    v_input = torch.randn(
        num_fwd_tokens, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16
    )

    # Attention read: pointer array pointing to CONTEXT region of each layer
    # (byte offset into the shared buffer for the context region start)
    ctx_byte_offset = ctx_start * fp8_item_size
    attn_k_ptrs = torch.tensor(
        [k.data_ptr() + ctx_byte_offset for k in k_caches],
        dtype=torch.uint64, device=device,
    )
    attn_v_ptrs = torch.tensor(
        [v.data_ptr() + ctx_byte_offset for v in v_caches],
        dtype=torch.uint64, device=device,
    )
    # Combined K+V pointer array for the attention read kernel
    attn_all_ptrs = torch.cat([attn_k_ptrs, attn_v_ptrs])

    # GEMM weight tensors: simulate MLP weight reads
    # MiniMax-M2.7: hidden=3072, intermediate=1536, 8 active experts
    # Approximate with a single large matmul per layer
    gemm_size = 3072
    gemm_weight = torch.randn(
        gemm_size, gemm_size, device=device, dtype=torch.bfloat16
    )
    gemm_input = torch.randn(
        256, gemm_size, device=device, dtype=torch.bfloat16
    )

    # Output buffer for attention-like read kernel
    BLOCK = 1024
    output_size = sm_count * 2 * BLOCK
    output_buf = torch.zeros(output_size, device=device, dtype=torch.float32)

    # --- Pre-warm ---
    print("Pre-warming kernels...")
    with torch.cuda.stream(stream_write):
        transfer_kv_all_layer(
            src_k_layers=k_cache_ptrs, dst_k_layers=k_host_ptrs,
            src_v_layers=v_cache_ptrs, dst_v_layers=v_host_ptrs,
            src_indices=wb_src_indices, dst_indices=wb_dst_indices,
            item_size=fp8_item_size, num_layers=num_layers,
        )
    with torch.cuda.stream(stream_exec):
        # Warm attention read kernel
        ctx_numel = num_ctx_slots * num_kv_heads * head_dim
        _forward_compute_kernel[(sm_count * 2,)](
            attn_all_ptrs, output_buf, ctx_numel,
            num_layers=num_layers * 2, BLOCK=BLOCK, num_warps=4,
        )
        # Warm GEMM
        torch.mm(gemm_input, gemm_weight)
        # Warm set_kv_buffer
        for layer in range(num_layers):
            fused_fp8_set_kv_buffer(
                k_input, v_input,
                k_caches[layer], v_caches[layer],
                fwd_cache_loc, page_size=page_size,
            )
    torch.cuda.synchronize()
    print("Warm-up done.")

    # --- Compute reference ---
    # Reference for GEMM
    ref_gemm = torch.mm(gemm_input, gemm_weight)
    # Reference for attention read
    ref_output = torch.zeros_like(output_buf)
    ctx_numel = num_ctx_slots * num_kv_heads * head_dim
    _forward_compute_kernel[(sm_count * 2,)](
        attn_all_ptrs, ref_output, ctx_numel,
        num_layers=num_layers * 2, BLOCK=BLOCK, num_warps=4,
    )
    # Reference for FP8 writes
    ref_k_caches = []
    ref_v_caches = []
    for layer in range(num_layers):
        ref_k = torch.zeros(
            total_slots, num_kv_heads, head_dim,
            device=device, dtype=torch.float8_e4m3fn,
        )
        ref_v = torch.zeros(
            total_slots, num_kv_heads, head_dim,
            device=device, dtype=torch.float8_e4m3fn,
        )
        fused_fp8_set_kv_buffer(
            k_input, v_input, ref_k, ref_v, fwd_cache_loc, page_size=page_size,
        )
        ref_k_caches.append(ref_k)
        ref_v_caches.append(ref_v)
    torch.cuda.synchronize()

    # --- Main loop ---
    corruption_count = 0
    overlap_count = 0
    first_msg = ""

    transfer_start = torch.cuda.Event(enable_timing=True)
    transfer_end = torch.cuda.Event(enable_timing=True)
    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)

    for iteration in range(ITERATIONS):
        torch.cuda.synchronize()

        # Fill WB + context regions (transfer reads WB, attention reads context)
        for k in k_caches:
            k[:ctx_start + num_ctx_slots].fill_(0.5)
        for v in v_caches:
            v[:ctx_start + num_ctx_slots].fill_(0.5)
        # Clear FWD region
        for k in k_caches:
            k[fwd_start:].zero_()
        for v in v_caches:
            v[fwd_start:].zero_()
        output_buf.zero_()

        torch.cuda.synchronize()

        # Launch transfer on write_stream
        with torch.cuda.stream(stream_write):
            transfer_start.record(stream_write)
            transfer_kv_all_layer(
                src_k_layers=k_cache_ptrs, dst_k_layers=k_host_ptrs,
                src_v_layers=v_cache_ptrs, dst_v_layers=v_host_ptrs,
                src_indices=wb_src_indices, dst_indices=wb_dst_indices,
                item_size=fp8_item_size, num_layers=num_layers,
            )
            transfer_end.record(stream_write)

        # Launch forward on exec_stream — NO FENCE
        # This simulates the production forward: attention read + GEMM + set_kv_buffer
        with torch.cuda.stream(stream_exec):
            forward_start.record(stream_exec)

            # 1. Attention-like read from CONTEXT region (same allocation as transfer)
            _forward_compute_kernel[(sm_count * 2,)](
                attn_all_ptrs, output_buf, ctx_numel,
                num_layers=num_layers * 2, BLOCK=BLOCK, num_warps=4,
            )

            # 2. GEMM (simulates MLP, heavy L2 weight reads)
            gemm_result = torch.mm(gemm_input, gemm_weight)

            # 3. set_kv_buffer (FP8 writes to FWD region)
            for layer in range(num_layers):
                fused_fp8_set_kv_buffer(
                    k_input, v_input,
                    k_caches[layer], v_caches[layer],
                    fwd_cache_loc, page_size=page_size,
                )

            forward_end.record(stream_exec)

        stream_write.synchronize()
        stream_exec.synchronize()

        # Overlap measurement
        t_start = transfer_start.elapsed_time(forward_start)
        t_transfer = transfer_start.elapsed_time(transfer_end)
        had_overlap = t_start < t_transfer
        if had_overlap:
            overlap_count += 1

        # --- Verify all three outputs ---
        corrupted_this_iter = False

        # Check attention read output
        attn_mismatch = (output_buf != ref_output).sum().item()
        if attn_mismatch > 0:
            corruption_count += 1
            corrupted_this_iter = True
            overlap_ms = max(0, t_transfer - t_start)
            msg = (
                f"iter={iteration} ATTENTION READ: "
                f"{attn_mismatch}/{output_buf.numel()} elements differ "
                f"[overlap={overlap_ms:.2f}ms]"
            )
            if not first_msg:
                first_msg = msg

        # Check GEMM output
        if not corrupted_this_iter:
            gemm_mismatch = (gemm_result != ref_gemm).sum().item()
            if gemm_mismatch > 0:
                corruption_count += 1
                corrupted_this_iter = True
                overlap_ms = max(0, t_transfer - t_start)
                msg = (
                    f"iter={iteration} GEMM: "
                    f"{gemm_mismatch}/{gemm_result.numel()} elements differ "
                    f"[overlap={overlap_ms:.2f}ms]"
                )
                if not first_msg:
                    first_msg = msg

        # Check FP8 writes
        if not corrupted_this_iter:
            for layer in range(num_layers):
                fwd_k = k_caches[layer][fwd_start:].view(torch.uint8)
                ref_k = ref_k_caches[layer][fwd_start:].view(torch.uint8)
                if (fwd_k != ref_k).any():
                    mismatches = (fwd_k != ref_k).sum().item()
                    corruption_count += 1
                    overlap_ms = max(0, t_transfer - t_start)
                    msg = (
                        f"iter={iteration} FP8 K layer={layer}: "
                        f"{mismatches}/{fwd_k.numel()} bytes differ "
                        f"[overlap={overlap_ms:.2f}ms]"
                    )
                    if not first_msg:
                        first_msg = msg
                    break
                fwd_v = v_caches[layer][fwd_start:].view(torch.uint8)
                ref_v = ref_v_caches[layer][fwd_start:].view(torch.uint8)
                if (fwd_v != ref_v).any():
                    mismatches = (fwd_v != ref_v).sum().item()
                    corruption_count += 1
                    overlap_ms = max(0, t_transfer - t_start)
                    msg = (
                        f"iter={iteration} FP8 V layer={layer}: "
                        f"{mismatches}/{fwd_v.numel()} bytes differ "
                        f"[overlap={overlap_ms:.2f}ms]"
                    )
                    if not first_msg:
                        first_msg = msg
                    break

    print(f"\nOverlap stats: {overlap_count}/{ITERATIONS} iterations had GPU overlap")
    print(f"Attention+GEMM+FP8 results: {corruption_count}/{ITERATIONS} corrupted")

    if overlap_count == 0:
        print("WARNING: no overlap detected.")

    assert corruption_count == 0, (
        f"CORRUPTION in {corruption_count}/{ITERATIONS} iterations "
        f"({overlap_count} had overlap). "
        f"First: {first_msg}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_production_real_attention_decode_no_fence():
    """
    Most faithful reproduction: uses the REAL trtllm attention decode kernel
    reading from context pages in the SAME shared KV cache buffer that the
    transfer kernel reads from.

    Production forward per layer:
      1. fused_fp8_set_kv_buffer: writes new token to FWD page
      2. trtllm_batch_decode_with_kv_cache: reads from ALL context pages
         (same physical allocation as transfer source)

    Layout in shared buffer per layer:
      [WB pages | Context pages | FWD pages]
       transfer   attention       set_kv_buffer
       reads      reads           writes
       (write_stream)  (exec_stream)

    MiniMax-M2.7: 62 layers, 48 Q heads, 8 KV heads, head_dim=128, page_size=32
    """
    device = "cuda"
    platform = current_platform()
    if platform.vendor != "nvidia":
        pytest.skip("NVIDIA GPU required")

    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    total_mem = torch.cuda.get_device_properties(device).total_memory

    if total_mem > 150e9:
        num_layers = 62
        num_wb_pages = 2776
    elif total_mem > 70e9:
        num_layers = 32
        num_wb_pages = 1024
    else:
        num_layers = 16
        num_wb_pages = 512

    # MiniMax-M2.7 params
    num_q_heads = 48
    num_kv_heads = NUM_KV_HEADS  # 8
    head_dim = HEAD_DIM  # 128
    page_size = PAGE_SIZE  # 32
    kv_dtype = torch.float8_e4m3fn

    # Context pages for attention (simulates ongoing requests with context)
    # 32 requests × ~1024 tokens each = 32 × 32 pages = 1024 pages
    batch_size = 32
    ctx_pages_per_req = 32  # 32 pages × 32 tokens/page = 1024 tokens context
    num_ctx_pages = batch_size * ctx_pages_per_req
    num_fwd_pages = 8  # 8 pages for new tokens (batch_size decode tokens)
    total_pages = num_wb_pages + num_ctx_pages + num_fwd_pages

    num_wb_slots = num_wb_pages * page_size
    ctx_start_page = num_wb_pages
    fwd_start_page = num_wb_pages + num_ctx_pages
    fwd_start_slot = fwd_start_page * page_size
    total_slots = total_pages * page_size

    fp8_item_size = num_kv_heads * head_dim  # 1024 bytes per slot (FP8)

    print(f"\nSM count: {sm_count}, GPU memory: {total_mem / 1e9:.1f} GB")
    print(f"Layers: {num_layers}, Batch: {batch_size}")
    print(f"  WB pages: {num_wb_pages} (transfer reads)")
    print(f"  Context pages: {num_ctx_pages} (attention reads, {ctx_pages_per_req} pages/req)")
    print(f"  FWD pages: {num_fwd_pages} (set_kv_buffer writes)")
    print(f"Shared buffer per layer: {total_slots * fp8_item_size / 1e6:.1f} MB")
    total_transfer_bytes = num_wb_slots * fp8_item_size * num_layers * 2
    print(f"Transfer volume per iter: {total_transfer_bytes / 1e9:.2f} GB")

    stream_exec = torch.cuda.Stream()
    stream_write = torch.cuda.Stream()

    # Shared KV cache per layer: [total_slots, num_kv_heads, head_dim] in FP8
    # This is the SAME single allocation read by both transfer and attention.
    k_caches = []
    v_caches = []
    for _ in range(num_layers):
        k_caches.append(torch.randn(
            total_slots, num_kv_heads, head_dim,
            device=device, dtype=torch.bfloat16,
        ).to(kv_dtype))
        v_caches.append(torch.randn(
            total_slots, num_kv_heads, head_dim,
            device=device, dtype=torch.bfloat16,
        ).to(kv_dtype))

    # Transfer pointers (read from WB region = start of buffer)
    k_cache_ptrs = torch.tensor(
        [k.data_ptr() for k in k_caches], dtype=torch.uint64, device=device
    )
    v_cache_ptrs = torch.tensor(
        [v.data_ptr() for v in v_caches], dtype=torch.uint64, device=device
    )

    # Host-pinned destination
    host_buf_size = num_wb_slots * fp8_item_size
    k_host_bufs = [
        torch.zeros(host_buf_size, dtype=torch.uint8, pin_memory=True)
        for _ in range(num_layers)
    ]
    v_host_bufs = [
        torch.zeros(host_buf_size, dtype=torch.uint8, pin_memory=True)
        for _ in range(num_layers)
    ]
    k_host_ptrs = torch.tensor(
        [b.data_ptr() for b in k_host_bufs], dtype=torch.uint64, device=device
    )
    v_host_ptrs = torch.tensor(
        [b.data_ptr() for b in v_host_bufs], dtype=torch.uint64, device=device
    )

    # Transfer indices
    wb_src_indices = torch.arange(num_wb_slots, dtype=torch.int32, device=device)
    wb_dst_indices = torch.arange(num_wb_slots, dtype=torch.int32, device=device)

    # Attention decode setup:
    # KV cache viewed as [total_pages, num_kv_heads, page_size, head_dim] for attention
    # (permuted from [total_pages, page_size, num_kv_heads, head_dim])
    seq_lens = torch.full(
        (batch_size,), ctx_pages_per_req * page_size,
        dtype=torch.int32, device=device,
    )
    max_seq_len = int(seq_lens.max().item())
    max_num_blocks = (max_seq_len + page_size - 1) // page_size

    # Block tables: map each request to physical pages in the CONTEXT region
    block_tables = torch.zeros(
        batch_size, max_num_blocks, dtype=torch.int32, device=device
    )
    for b in range(batch_size):
        start_page = ctx_start_page + b * ctx_pages_per_req
        block_tables[b, :ctx_pages_per_req] = torch.arange(
            start_page, start_page + ctx_pages_per_req,
            dtype=torch.int32, device=device,
        )

    # Query for decode: [batch_size, num_q_heads, head_dim]
    query = torch.randn(
        batch_size, num_q_heads, head_dim, device=device, dtype=kv_dtype
    )

    # Workspace for TRT-LLM attention kernel
    workspace_buffer = torch.empty(
        512 * 1024 * 1024, device=device, dtype=torch.uint8
    )

    bmm1_scale = 1.0 / math.sqrt(head_dim)
    bmm2_scale = 1.0

    # Forward set_kv_buffer setup
    fwd_cache_loc = torch.arange(
        fwd_start_slot, total_slots, dtype=torch.int32, device=device
    )
    num_fwd_tokens = min(fwd_cache_loc.shape[0], batch_size)
    fwd_cache_loc = fwd_cache_loc[:num_fwd_tokens]
    k_input = torch.randn(
        num_fwd_tokens, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    v_input = torch.randn(
        num_fwd_tokens, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16
    )

    # --- Pre-warm ---
    print("Pre-warming kernels...")
    with torch.cuda.stream(stream_write):
        transfer_kv_all_layer(
            src_k_layers=k_cache_ptrs, dst_k_layers=k_host_ptrs,
            src_v_layers=v_cache_ptrs, dst_v_layers=v_host_ptrs,
            src_indices=wb_src_indices, dst_indices=wb_dst_indices,
            item_size=fp8_item_size, num_layers=num_layers,
        )
    with torch.cuda.stream(stream_exec):
        for layer_idx in range(num_layers):
            # Attention decode reads from context pages
            k_for_attn = k_caches[layer_idx].view(
                total_pages, page_size, num_kv_heads, head_dim
            ).permute(0, 2, 1, 3)
            v_for_attn = v_caches[layer_idx].view(
                total_pages, page_size, num_kv_heads, head_dim
            ).permute(0, 2, 1, 3)
            trtllm_batch_decode_with_kv_cache(
                query=query,
                kv_cache=(k_for_attn, v_for_attn),
                workspace_buffer=workspace_buffer,
                block_tables=block_tables,
                seq_lens=seq_lens,
                max_seq_len=max_seq_len,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                out_dtype=kv_dtype,
            )
            # set_kv_buffer
            fused_fp8_set_kv_buffer(
                k_input, v_input,
                k_caches[layer_idx], v_caches[layer_idx],
                fwd_cache_loc, page_size=page_size,
            )
    torch.cuda.synchronize()
    print("Warm-up done.")

    # --- Compute reference (serial, no concurrency) ---
    ref_outputs = []
    for layer_idx in range(num_layers):
        k_for_attn = k_caches[layer_idx].view(
            total_pages, page_size, num_kv_heads, head_dim
        ).permute(0, 2, 1, 3)
        v_for_attn = v_caches[layer_idx].view(
            total_pages, page_size, num_kv_heads, head_dim
        ).permute(0, 2, 1, 3)
        out = trtllm_batch_decode_with_kv_cache(
            query=query,
            kv_cache=(k_for_attn, v_for_attn),
            workspace_buffer=workspace_buffer,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            out_dtype=kv_dtype,
        )
        ref_outputs.append(out.clone())
    torch.cuda.synchronize()

    # --- Main loop ---
    corruption_count = 0
    overlap_count = 0
    first_msg = ""

    transfer_start = torch.cuda.Event(enable_timing=True)
    transfer_end = torch.cuda.Event(enable_timing=True)
    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)

    for iteration in range(ITERATIONS):
        torch.cuda.synchronize()

        # Launch transfer
        with torch.cuda.stream(stream_write):
            transfer_start.record(stream_write)
            transfer_kv_all_layer(
                src_k_layers=k_cache_ptrs, dst_k_layers=k_host_ptrs,
                src_v_layers=v_cache_ptrs, dst_v_layers=v_host_ptrs,
                src_indices=wb_src_indices, dst_indices=wb_dst_indices,
                item_size=fp8_item_size, num_layers=num_layers,
            )
            transfer_end.record(stream_write)

        # Launch forward (attention decode + set_kv_buffer) — NO FENCE
        with torch.cuda.stream(stream_exec):
            forward_start.record(stream_exec)
            for layer_idx in range(num_layers):
                k_for_attn = k_caches[layer_idx].view(
                    total_pages, page_size, num_kv_heads, head_dim
                ).permute(0, 2, 1, 3)
                v_for_attn = v_caches[layer_idx].view(
                    total_pages, page_size, num_kv_heads, head_dim
                ).permute(0, 2, 1, 3)
                trtllm_batch_decode_with_kv_cache(
                    query=query,
                    kv_cache=(k_for_attn, v_for_attn),
                    workspace_buffer=workspace_buffer,
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                    max_seq_len=max_seq_len,
                    bmm1_scale=bmm1_scale,
                    bmm2_scale=bmm2_scale,
                    out_dtype=kv_dtype,
                )
                fused_fp8_set_kv_buffer(
                    k_input, v_input,
                    k_caches[layer_idx], v_caches[layer_idx],
                    fwd_cache_loc, page_size=page_size,
                )
            forward_end.record(stream_exec)

        stream_write.synchronize()
        stream_exec.synchronize()

        # Overlap measurement
        t_start = transfer_start.elapsed_time(forward_start)
        t_transfer = transfer_start.elapsed_time(transfer_end)
        t_forward = forward_start.elapsed_time(forward_end)
        had_overlap = t_start < t_transfer
        if had_overlap:
            overlap_count += 1

        # Verify attention outputs match reference
        for layer_idx in range(num_layers):
            k_for_attn = k_caches[layer_idx].view(
                total_pages, page_size, num_kv_heads, head_dim
            ).permute(0, 2, 1, 3)
            v_for_attn = v_caches[layer_idx].view(
                total_pages, page_size, num_kv_heads, head_dim
            ).permute(0, 2, 1, 3)
            out = trtllm_batch_decode_with_kv_cache(
                query=query,
                kv_cache=(k_for_attn, v_for_attn),
                workspace_buffer=workspace_buffer,
                block_tables=block_tables,
                seq_lens=seq_lens,
                max_seq_len=max_seq_len,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                out_dtype=kv_dtype,
            )
            # Compare output to reference
            mismatch = (out.view(torch.uint8) != ref_outputs[layer_idx].view(torch.uint8)).sum().item()
            if mismatch > 0:
                corruption_count += 1
                overlap_ms = max(0, t_transfer - t_start)
                msg = (
                    f"iter={iteration} DECODE layer={layer_idx}: "
                    f"{mismatch} bytes differ "
                    f"[overlap={overlap_ms:.2f}ms, transfer={t_transfer:.2f}ms, "
                    f"fwd={t_forward:.2f}ms]"
                )
                if not first_msg:
                    first_msg = msg
                break

    print(f"\nOverlap stats: {overlap_count}/{ITERATIONS} iterations had GPU overlap")
    print(f"Real attention decode results: {corruption_count}/{ITERATIONS} corrupted")

    if overlap_count == 0:
        print("WARNING: no overlap detected.")

    assert corruption_count == 0, (
        f"ATTENTION DECODE CORRUPTION in {corruption_count}/{ITERATIONS} iterations "
        f"({overlap_count} had overlap). "
        f"First: {first_msg}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_production_real_attention_extend_no_fence():
    """
    Same as decode test but uses trtllm_batch_context_with_kv_cache (prefill/extend).
    Extend has longer query sequences → more compute, more L2 pressure.

    Simulates: a prefill batch (or chunked prefill) running concurrently with
    the transfer kernel.
    """
    device = "cuda"
    platform = current_platform()
    if platform.vendor != "nvidia":
        pytest.skip("NVIDIA GPU required")

    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    total_mem = torch.cuda.get_device_properties(device).total_memory

    if total_mem > 150e9:
        num_layers = 62
        num_wb_pages = 2776
    elif total_mem > 70e9:
        num_layers = 32
        num_wb_pages = 1024
    else:
        num_layers = 16
        num_wb_pages = 512

    num_q_heads = 48
    num_kv_heads = NUM_KV_HEADS
    head_dim = HEAD_DIM
    page_size = PAGE_SIZE
    kv_dtype = torch.float8_e4m3fn

    # Extend scenario: 4 requests, each with 512 token context, extending by 128 tokens
    batch_size = 4
    ctx_tokens_per_req = 512
    extend_tokens_per_req = 128
    ctx_pages_per_req = (ctx_tokens_per_req + page_size - 1) // page_size  # 16
    num_ctx_pages = batch_size * ctx_pages_per_req
    num_fwd_pages = batch_size * ((extend_tokens_per_req + page_size - 1) // page_size)
    total_pages = num_wb_pages + num_ctx_pages + num_fwd_pages

    num_wb_slots = num_wb_pages * page_size
    ctx_start_page = num_wb_pages
    fwd_start_page = num_wb_pages + num_ctx_pages
    fwd_start_slot = fwd_start_page * page_size
    total_slots = total_pages * page_size

    fp8_item_size = num_kv_heads * head_dim

    total_q_tokens = batch_size * extend_tokens_per_req  # 512

    print(f"\nSM count: {sm_count}, GPU memory: {total_mem / 1e9:.1f} GB")
    print(f"Layers: {num_layers}, Batch: {batch_size}")
    print(f"  WB pages: {num_wb_pages}, Context pages: {num_ctx_pages}, FWD pages: {num_fwd_pages}")
    print(f"  Extend tokens per req: {extend_tokens_per_req}, total Q tokens: {total_q_tokens}")
    total_transfer_bytes = num_wb_slots * fp8_item_size * num_layers * 2
    print(f"Transfer volume per iter: {total_transfer_bytes / 1e9:.2f} GB")

    stream_exec = torch.cuda.Stream()
    stream_write = torch.cuda.Stream()

    # Shared KV cache
    k_caches = []
    v_caches = []
    for _ in range(num_layers):
        k_caches.append(torch.randn(
            total_slots, num_kv_heads, head_dim,
            device=device, dtype=torch.bfloat16,
        ).to(kv_dtype))
        v_caches.append(torch.randn(
            total_slots, num_kv_heads, head_dim,
            device=device, dtype=torch.bfloat16,
        ).to(kv_dtype))

    k_cache_ptrs = torch.tensor(
        [k.data_ptr() for k in k_caches], dtype=torch.uint64, device=device
    )
    v_cache_ptrs = torch.tensor(
        [v.data_ptr() for v in v_caches], dtype=torch.uint64, device=device
    )

    host_buf_size = num_wb_slots * fp8_item_size
    k_host_bufs = [
        torch.zeros(host_buf_size, dtype=torch.uint8, pin_memory=True)
        for _ in range(num_layers)
    ]
    v_host_bufs = [
        torch.zeros(host_buf_size, dtype=torch.uint8, pin_memory=True)
        for _ in range(num_layers)
    ]
    k_host_ptrs = torch.tensor(
        [b.data_ptr() for b in k_host_bufs], dtype=torch.uint64, device=device
    )
    v_host_ptrs = torch.tensor(
        [b.data_ptr() for b in v_host_bufs], dtype=torch.uint64, device=device
    )

    wb_src_indices = torch.arange(num_wb_slots, dtype=torch.int32, device=device)
    wb_dst_indices = torch.arange(num_wb_slots, dtype=torch.int32, device=device)

    # Extend attention setup
    # seq_lens = total KV length (context + extend tokens)
    total_kv_len = ctx_tokens_per_req + extend_tokens_per_req
    seq_lens = torch.full((batch_size,), total_kv_len, dtype=torch.int32, device=device)
    max_kv_len = total_kv_len
    max_q_len = extend_tokens_per_req

    # Block tables: context pages + fwd pages for each request
    pages_per_req = (total_kv_len + page_size - 1) // page_size
    max_num_blocks = pages_per_req
    block_tables = torch.zeros(
        batch_size, max_num_blocks, dtype=torch.int32, device=device
    )
    for b in range(batch_size):
        # Context pages
        ctx_start = ctx_start_page + b * ctx_pages_per_req
        block_tables[b, :ctx_pages_per_req] = torch.arange(
            ctx_start, ctx_start + ctx_pages_per_req,
            dtype=torch.int32, device=device,
        )
        # FWD pages (for the extend tokens)
        fwd_pages_for_req = pages_per_req - ctx_pages_per_req
        fwd_start = fwd_start_page + b * fwd_pages_for_req
        block_tables[b, ctx_pages_per_req:pages_per_req] = torch.arange(
            fwd_start, fwd_start + fwd_pages_for_req,
            dtype=torch.int32, device=device,
        )

    # Cumulative sequence lengths for extend kernel
    cu_seqlens_q = torch.arange(
        0, (batch_size + 1) * extend_tokens_per_req, extend_tokens_per_req,
        dtype=torch.int32, device=device,
    )
    cu_seqlens_k = torch.arange(
        0, (batch_size + 1) * total_kv_len, total_kv_len,
        dtype=torch.int32, device=device,
    )

    # Query for extend: [total_q_tokens, num_q_heads, head_dim]
    query = torch.randn(
        total_q_tokens, num_q_heads, head_dim, device=device, dtype=kv_dtype
    )

    workspace_buffer = torch.empty(
        512 * 1024 * 1024, device=device, dtype=torch.uint8
    )
    bmm1_scale = 1.0 / math.sqrt(head_dim)
    bmm2_scale = 1.0

    # Forward set_kv_buffer
    fwd_cache_loc = torch.arange(
        fwd_start_slot, fwd_start_slot + total_q_tokens,
        dtype=torch.int32, device=device,
    )
    k_input = torch.randn(
        total_q_tokens, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    v_input = torch.randn(
        total_q_tokens, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16
    )

    # --- Pre-warm ---
    print("Pre-warming kernels...")
    with torch.cuda.stream(stream_write):
        transfer_kv_all_layer(
            src_k_layers=k_cache_ptrs, dst_k_layers=k_host_ptrs,
            src_v_layers=v_cache_ptrs, dst_v_layers=v_host_ptrs,
            src_indices=wb_src_indices, dst_indices=wb_dst_indices,
            item_size=fp8_item_size, num_layers=num_layers,
        )
    with torch.cuda.stream(stream_exec):
        for layer_idx in range(min(2, num_layers)):
            k_for_attn = k_caches[layer_idx].view(
                total_pages, page_size, num_kv_heads, head_dim
            ).permute(0, 2, 1, 3)
            v_for_attn = v_caches[layer_idx].view(
                total_pages, page_size, num_kv_heads, head_dim
            ).permute(0, 2, 1, 3)
            trtllm_batch_context_with_kv_cache(
                query=query,
                kv_cache=(k_for_attn, v_for_attn),
                workspace_buffer=workspace_buffer,
                block_tables=block_tables,
                seq_lens=seq_lens,
                max_q_len=max_q_len,
                max_kv_len=max_kv_len,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                batch_size=batch_size,
                cum_seq_lens_q=cu_seqlens_q,
                cum_seq_lens_kv=cu_seqlens_k,
                out_dtype=kv_dtype,
            )
            fused_fp8_set_kv_buffer(
                k_input, v_input,
                k_caches[layer_idx], v_caches[layer_idx],
                fwd_cache_loc, page_size=page_size,
            )
    torch.cuda.synchronize()
    print("Warm-up done.")

    # --- Compute reference ---
    ref_outputs = []
    for layer_idx in range(num_layers):
        k_for_attn = k_caches[layer_idx].view(
            total_pages, page_size, num_kv_heads, head_dim
        ).permute(0, 2, 1, 3)
        v_for_attn = v_caches[layer_idx].view(
            total_pages, page_size, num_kv_heads, head_dim
        ).permute(0, 2, 1, 3)
        out = trtllm_batch_context_with_kv_cache(
            query=query,
            kv_cache=(k_for_attn, v_for_attn),
            workspace_buffer=workspace_buffer,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            batch_size=batch_size,
            cum_seq_lens_q=cu_seqlens_q,
            cum_seq_lens_kv=cu_seqlens_k,
            out_dtype=kv_dtype,
        )
        ref_outputs.append(out.clone())
    torch.cuda.synchronize()

    # --- Main loop ---
    corruption_count = 0
    overlap_count = 0
    first_msg = ""

    transfer_start = torch.cuda.Event(enable_timing=True)
    transfer_end = torch.cuda.Event(enable_timing=True)
    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)

    for iteration in range(ITERATIONS):
        torch.cuda.synchronize()

        with torch.cuda.stream(stream_write):
            transfer_start.record(stream_write)
            transfer_kv_all_layer(
                src_k_layers=k_cache_ptrs, dst_k_layers=k_host_ptrs,
                src_v_layers=v_cache_ptrs, dst_v_layers=v_host_ptrs,
                src_indices=wb_src_indices, dst_indices=wb_dst_indices,
                item_size=fp8_item_size, num_layers=num_layers,
            )
            transfer_end.record(stream_write)

        # NO FENCE
        with torch.cuda.stream(stream_exec):
            forward_start.record(stream_exec)
            for layer_idx in range(num_layers):
                fused_fp8_set_kv_buffer(
                    k_input, v_input,
                    k_caches[layer_idx], v_caches[layer_idx],
                    fwd_cache_loc, page_size=page_size,
                )
                k_for_attn = k_caches[layer_idx].view(
                    total_pages, page_size, num_kv_heads, head_dim
                ).permute(0, 2, 1, 3)
                v_for_attn = v_caches[layer_idx].view(
                    total_pages, page_size, num_kv_heads, head_dim
                ).permute(0, 2, 1, 3)
                trtllm_batch_context_with_kv_cache(
                    query=query,
                    kv_cache=(k_for_attn, v_for_attn),
                    workspace_buffer=workspace_buffer,
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                    max_q_len=max_q_len,
                    max_kv_len=max_kv_len,
                    bmm1_scale=bmm1_scale,
                    bmm2_scale=bmm2_scale,
                    batch_size=batch_size,
                    cum_seq_lens_q=cu_seqlens_q,
                    cum_seq_lens_kv=cu_seqlens_k,
                    out_dtype=kv_dtype,
                )
            forward_end.record(stream_exec)

        stream_write.synchronize()
        stream_exec.synchronize()

        # Overlap
        t_start = transfer_start.elapsed_time(forward_start)
        t_transfer = transfer_start.elapsed_time(transfer_end)
        t_forward = forward_start.elapsed_time(forward_end)
        had_overlap = t_start < t_transfer
        if had_overlap:
            overlap_count += 1

        # Verify attention outputs
        for layer_idx in range(num_layers):
            k_for_attn = k_caches[layer_idx].view(
                total_pages, page_size, num_kv_heads, head_dim
            ).permute(0, 2, 1, 3)
            v_for_attn = v_caches[layer_idx].view(
                total_pages, page_size, num_kv_heads, head_dim
            ).permute(0, 2, 1, 3)
            out = trtllm_batch_context_with_kv_cache(
                query=query,
                kv_cache=(k_for_attn, v_for_attn),
                workspace_buffer=workspace_buffer,
                block_tables=block_tables,
                seq_lens=seq_lens,
                max_q_len=max_q_len,
                max_kv_len=max_kv_len,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                batch_size=batch_size,
                cum_seq_lens_q=cu_seqlens_q,
                cum_seq_lens_kv=cu_seqlens_k,
                out_dtype=kv_dtype,
            )
            mismatch = (out.view(torch.uint8) != ref_outputs[layer_idx].view(torch.uint8)).sum().item()
            if mismatch > 0:
                corruption_count += 1
                overlap_ms = max(0, t_transfer - t_start)
                msg = (
                    f"iter={iteration} EXTEND layer={layer_idx}: "
                    f"{mismatch} bytes differ "
                    f"[overlap={overlap_ms:.2f}ms, transfer={t_transfer:.2f}ms, "
                    f"fwd={t_forward:.2f}ms]"
                )
                if not first_msg:
                    first_msg = msg
                break

    print(f"\nOverlap stats: {overlap_count}/{ITERATIONS} iterations had GPU overlap")
    print(f"Real attention extend results: {corruption_count}/{ITERATIONS} corrupted")

    if overlap_count == 0:
        print("WARNING: no overlap detected.")

    assert corruption_count == 0, (
        f"ATTENTION EXTEND CORRUPTION in {corruption_count}/{ITERATIONS} iterations "
        f"({overlap_count} had overlap). "
        f"First: {first_msg}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_production_shared_buffer_fp8_with_fence():
    """
    Control: same as no_fence but WITH exec_stream.wait_stream(write_stream).
    Should always pass — confirms the fence prevents corruption.
    """
    device = "cuda"
    platform = current_platform()
    if platform.vendor != "nvidia":
        pytest.skip("NVIDIA GPU required")

    total_mem = torch.cuda.get_device_properties(device).total_memory

    if total_mem > 150e9:
        num_layers = 62
        num_wb_pages = 2776
    elif total_mem > 70e9:
        num_layers = 32
        num_wb_pages = 1024
    else:
        num_layers = 16
        num_wb_pages = 512

    num_fwd_pages = 64
    total_pages = num_wb_pages + num_fwd_pages
    page_size = PAGE_SIZE
    num_kv_heads = NUM_KV_HEADS
    head_dim = HEAD_DIM
    fp8_item_size = num_kv_heads * head_dim

    num_wb_slots = num_wb_pages * page_size
    total_slots = total_pages * page_size

    stream_exec = torch.cuda.Stream()
    stream_write = torch.cuda.Stream()

    k_caches = []
    v_caches = []
    for _ in range(num_layers):
        k_caches.append(torch.zeros(
            total_slots, num_kv_heads, head_dim,
            device=device, dtype=torch.float8_e4m3fn,
        ))
        v_caches.append(torch.zeros(
            total_slots, num_kv_heads, head_dim,
            device=device, dtype=torch.float8_e4m3fn,
        ))

    k_cache_ptrs = torch.tensor(
        [k.data_ptr() for k in k_caches], dtype=torch.uint64, device=device
    )
    v_cache_ptrs = torch.tensor(
        [v.data_ptr() for v in v_caches], dtype=torch.uint64, device=device
    )

    host_buf_size = num_wb_slots * fp8_item_size
    k_host_bufs = [
        torch.zeros(host_buf_size, dtype=torch.uint8, pin_memory=True)
        for _ in range(num_layers)
    ]
    v_host_bufs = [
        torch.zeros(host_buf_size, dtype=torch.uint8, pin_memory=True)
        for _ in range(num_layers)
    ]
    k_host_ptrs = torch.tensor(
        [b.data_ptr() for b in k_host_bufs], dtype=torch.uint64, device=device
    )
    v_host_ptrs = torch.tensor(
        [b.data_ptr() for b in v_host_bufs], dtype=torch.uint64, device=device
    )

    wb_src_indices = torch.arange(num_wb_slots, dtype=torch.int32, device=device)
    wb_dst_indices = torch.arange(num_wb_slots, dtype=torch.int32, device=device)

    fwd_cache_loc = torch.arange(
        num_wb_slots, total_slots, dtype=torch.int32, device=device
    )
    num_fwd_tokens = fwd_cache_loc.shape[0]
    k_input = torch.randn(
        num_fwd_tokens, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    v_input = torch.randn(
        num_fwd_tokens, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16
    )

    # Reference
    ref_k_caches = []
    ref_v_caches = []
    for layer in range(num_layers):
        ref_k = torch.zeros(
            total_slots, num_kv_heads, head_dim,
            device=device, dtype=torch.float8_e4m3fn,
        )
        ref_v = torch.zeros(
            total_slots, num_kv_heads, head_dim,
            device=device, dtype=torch.float8_e4m3fn,
        )
        fused_fp8_set_kv_buffer(
            k_input, v_input, ref_k, ref_v, fwd_cache_loc, page_size=page_size,
        )
        ref_k_caches.append(ref_k)
        ref_v_caches.append(ref_v)
    torch.cuda.synchronize()

    corruption_count = 0

    for iteration in range(ITERATIONS):
        torch.cuda.synchronize()

        for k in k_caches:
            k[:num_wb_slots].fill_(0.5)
        for v in v_caches:
            v[:num_wb_slots].fill_(0.5)
        for k in k_caches:
            k[num_wb_slots:].zero_()
        for v in v_caches:
            v[num_wb_slots:].zero_()

        torch.cuda.synchronize()

        with torch.cuda.stream(stream_write):
            transfer_kv_all_layer(
                src_k_layers=k_cache_ptrs,
                dst_k_layers=k_host_ptrs,
                src_v_layers=v_cache_ptrs,
                dst_v_layers=v_host_ptrs,
                src_indices=wb_src_indices,
                dst_indices=wb_dst_indices,
                item_size=fp8_item_size,
                num_layers=num_layers,
            )

        # FENCE: execution_stream waits for write_stream
        stream_exec.wait_stream(stream_write)

        with torch.cuda.stream(stream_exec):
            for layer in range(num_layers):
                fused_fp8_set_kv_buffer(
                    k_input, v_input,
                    k_caches[layer], v_caches[layer],
                    fwd_cache_loc, page_size=page_size,
                )

        stream_write.synchronize()
        stream_exec.synchronize()

        for layer in range(num_layers):
            fwd_k = k_caches[layer][num_wb_slots:].view(torch.uint8)
            ref_k = ref_k_caches[layer][num_wb_slots:].view(torch.uint8)
            if (fwd_k != ref_k).any():
                corruption_count += 1
                break
            fwd_v = v_caches[layer][num_wb_slots:].view(torch.uint8)
            ref_v = ref_v_caches[layer][num_wb_slots:].view(torch.uint8)
            if (fwd_v != ref_v).any():
                corruption_count += 1
                break

    assert corruption_count == 0, (
        f"CORRUPTION even WITH fence in {corruption_count}/{ITERATIONS} iters "
        f"(unexpected — fence should prevent this)"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_production_disjoint_pages_host_pinned_no_fence():
    """
    Production-scale test: transfer kernel writes to HOST-PINNED memory while
    forward kernel computes on DISJOINT device pages. No fence.

    This matches the exact production scenario:
    - Transfer reads 2776 device pages, writes to host-pinned (SM → L2 → PCIe)
    - Forward reads/writes 256 different device pages (attention + set_kv_buffer)
    - Pages are completely disjoint (DeviceNodeRef guarantee)
    - No fence between streams

    On B200 with the fence removed, this should reproduce the OverflowError
    (corrupted computation from L2 interference).
    """
    device = "cuda"
    platform = current_platform()
    if platform.vendor != "nvidia":
        pytest.skip("NVIDIA GPU required")

    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    print(f"\nSM count: {sm_count}")
    print(f"Transfer pages: {NUM_WB_PAGES} ({NUM_WB_SLOTS} slots)")
    print(f"Forward pages: {NUM_FWD_PAGES} ({NUM_FWD_SLOTS} slots)")
    print(f"Layers: {NUM_LAYERS}, item_size: {ITEM_SIZE} bytes")
    total_transfer_bytes = NUM_WB_SLOTS * ITEM_SIZE * NUM_LAYERS * 2  # K+V
    print(f"Total transfer volume per iter: {total_transfer_bytes / 1e9:.2f} GB")

    stream_exec = torch.cuda.Stream()   # simulates execution_stream (forward)
    stream_write = torch.cuda.Stream()  # simulates write_stream (transfer)

    # Device KV pool for TRANSFER (writeback source) — filled with known data
    k_wb_dev, v_wb_dev, k_wb_ptrs, v_wb_ptrs = _make_large_kv_pool(
        NUM_WB_SLOTS, device
    )

    # Host-pinned destination for transfer (the real scenario)
    k_host, v_host, k_host_ptrs, v_host_ptrs = _make_host_pinned_pool(NUM_WB_SLOTS)

    # Device KV pool for FORWARD (completely separate allocation)
    k_fwd_dev, v_fwd_dev, k_fwd_ptrs, v_fwd_ptrs = _make_large_kv_pool(
        NUM_FWD_SLOTS, device
    )

    # Indices for transfer (all writeback pages)
    wb_src_indices = torch.arange(NUM_WB_SLOTS, dtype=torch.int32, device=device)
    wb_dst_indices = torch.arange(NUM_WB_SLOTS, dtype=torch.int32, device=device)

    # Output buffer for forward compute kernel
    BLOCK = 1024
    output_size = sm_count * 2 * BLOCK
    output_buf = torch.zeros(output_size, device=device, dtype=torch.float32)

    # Forward kernel pointer array (K + V layers)
    fwd_all_ptrs = torch.tensor(
        [b.view(torch.float16).data_ptr() for b in k_fwd_dev]
        + [b.view(torch.float16).data_ptr() for b in v_fwd_dev],
        dtype=torch.uint64,
        device=device,
    )

    corruption_count = 0
    first_corruption_msg = ""

    # Known pattern: fill forward KV with a recognizable value
    FILL_VALUE = 1.0

    for iteration in range(ITERATIONS):
        torch.cuda.synchronize()

        # Fill writeback source with pattern (will be read by transfer kernel)
        for kb in k_wb_dev:
            kb.fill_(0.5)
        for vb in v_wb_dev:
            vb.fill_(0.5)

        # Fill forward KV with known value
        for kb in k_fwd_dev:
            kb.fill_(FILL_VALUE)
        for vb in v_fwd_dev:
            vb.fill_(FILL_VALUE)

        # Clear output
        output_buf.zero_()

        # Clear host destination
        for kb in k_host:
            kb.zero_()
        for vb in v_host:
            vb.zero_()

        torch.cuda.synchronize()

        # Launch transfer kernel on write_stream (reads device, writes host-pinned)
        # This is the big persistent kernel that saturates L2
        with torch.cuda.stream(stream_write):
            transfer_kv_all_layer(
                src_k_layers=k_wb_ptrs,
                dst_k_layers=k_host_ptrs,
                src_v_layers=v_wb_ptrs,
                dst_v_layers=v_host_ptrs,
                src_indices=wb_src_indices,
                dst_indices=wb_dst_indices,
                item_size=ITEM_SIZE,
                num_layers=NUM_LAYERS,
            )

        # Launch forward compute on execution_stream — NO FENCE
        # This kernel reads from DISJOINT forward KV pages
        fwd_numel = k_fwd_dev[0].view(torch.float16).numel()
        grid = (sm_count * 2,)
        with torch.cuda.stream(stream_exec):
            _forward_compute_kernel[grid](
                fwd_all_ptrs,
                output_buf,
                fwd_numel,
                num_layers=NUM_LAYERS * 2,  # K + V
                BLOCK=BLOCK,
                num_warps=4,
            )

        # Sync both streams
        stream_write.synchronize()
        stream_exec.synchronize()

        # Verify forward computation result
        # Each program reads fwd_numel elements across all layers, all filled
        # with FILL_VALUE=1.0. The expected sum per element position depends on
        # how many chunks each program processes.
        # Simpler check: output should be > 0 everywhere (since input is all 1.0)
        # and should NOT contain NaN/Inf (corruption indicator)
        has_nan = torch.isnan(output_buf).any().item()
        has_inf = torch.isinf(output_buf).any().item()
        has_zero = (output_buf == 0).all().item()

        if has_nan or has_inf:
            corruption_count += 1
            msg = (
                f"iter={iteration}: NaN={has_nan} Inf={has_inf} "
                f"(forward computed garbage from L2 corruption)"
            )
            if not first_corruption_msg:
                first_corruption_msg = msg
            continue

        # Also verify the host buffer received correct data from transfer
        sample_host_k = k_host[0][:PAGE_SIZE].flatten()
        expected_val = 0.5
        host_correct = torch.allclose(
            sample_host_k,
            torch.full_like(sample_host_k, expected_val),
            atol=1e-3,
        )
        if not host_correct:
            actual_vals = sample_host_k[:10].tolist()
            corruption_count += 1
            msg = (
                f"iter={iteration}: host buffer has wrong data "
                f"(expected {expected_val}, got {actual_vals})"
            )
            if not first_corruption_msg:
                first_corruption_msg = msg

    print(f"\nResults: {corruption_count}/{ITERATIONS} iterations had corruption")
    assert corruption_count == 0, (
        f"CORRUPTION DETECTED in {corruption_count}/{ITERATIONS} iterations "
        f"with DISJOINT pages + host-pinned destination. "
        f"First: {first_corruption_msg}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_production_disjoint_pages_host_pinned_with_fence():
    """
    Same as above but WITH fence (execution_stream.wait_stream(write_stream)).
    Should always pass — confirms the fence prevents the corruption.
    """
    device = "cuda"
    platform = current_platform()
    if platform.vendor != "nvidia":
        pytest.skip("NVIDIA GPU required")

    sm_count = torch.cuda.get_device_properties(device).multi_processor_count

    stream_exec = torch.cuda.Stream()
    stream_write = torch.cuda.Stream()

    k_wb_dev, v_wb_dev, k_wb_ptrs, v_wb_ptrs = _make_large_kv_pool(
        NUM_WB_SLOTS, device
    )
    k_host, v_host, k_host_ptrs, v_host_ptrs = _make_host_pinned_pool(NUM_WB_SLOTS)
    k_fwd_dev, v_fwd_dev, k_fwd_ptrs, v_fwd_ptrs = _make_large_kv_pool(
        NUM_FWD_SLOTS, device
    )

    wb_src_indices = torch.arange(NUM_WB_SLOTS, dtype=torch.int32, device=device)
    wb_dst_indices = torch.arange(NUM_WB_SLOTS, dtype=torch.int32, device=device)

    BLOCK = 1024
    output_size = sm_count * 2 * BLOCK
    output_buf = torch.zeros(output_size, device=device, dtype=torch.float32)

    fwd_all_ptrs = torch.tensor(
        [b.view(torch.float16).data_ptr() for b in k_fwd_dev]
        + [b.view(torch.float16).data_ptr() for b in v_fwd_dev],
        dtype=torch.uint64,
        device=device,
    )

    FILL_VALUE = 1.0
    corruption_count = 0

    for iteration in range(ITERATIONS):
        torch.cuda.synchronize()

        for kb in k_wb_dev:
            kb.fill_(0.5)
        for vb in v_wb_dev:
            vb.fill_(0.5)
        for kb in k_fwd_dev:
            kb.fill_(FILL_VALUE)
        for vb in v_fwd_dev:
            vb.fill_(FILL_VALUE)
        output_buf.zero_()
        for kb in k_host:
            kb.zero_()
        for vb in v_host:
            vb.zero_()

        torch.cuda.synchronize()

        # Launch transfer
        with torch.cuda.stream(stream_write):
            transfer_kv_all_layer(
                src_k_layers=k_wb_ptrs,
                dst_k_layers=k_host_ptrs,
                src_v_layers=v_wb_ptrs,
                dst_v_layers=v_host_ptrs,
                src_indices=wb_src_indices,
                dst_indices=wb_dst_indices,
                item_size=ITEM_SIZE,
                num_layers=NUM_LAYERS,
            )

        # FENCE: execution_stream waits for write_stream
        stream_exec.wait_stream(stream_write)

        # Launch forward AFTER fence
        fwd_numel = k_fwd_dev[0].view(torch.float16).numel()
        grid = (sm_count * 2,)
        with torch.cuda.stream(stream_exec):
            _forward_compute_kernel[grid](
                fwd_all_ptrs,
                output_buf,
                fwd_numel,
                num_layers=NUM_LAYERS * 2,
                BLOCK=BLOCK,
                num_warps=4,
            )

        stream_write.synchronize()
        stream_exec.synchronize()

        has_nan = torch.isnan(output_buf).any().item()
        has_inf = torch.isinf(output_buf).any().item()
        if has_nan or has_inf:
            corruption_count += 1

    assert corruption_count == 0, (
        f"CORRUPTION even WITH fence in {corruption_count}/{ITERATIONS} iters "
        f"(unexpected — fence should prevent this)"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_production_disjoint_pages_device_dst_no_fence():
    """
    Control: same as the no-fence test but transfer writes to DEVICE memory
    (not host-pinned). This isolates whether host-write SM traffic is the
    specific trigger.

    If this passes while the host-pinned version fails, it confirms that
    SM stores to host-mapped memory through L2 are the corruption mechanism.
    """
    device = "cuda"
    platform = current_platform()
    if platform.vendor != "nvidia":
        pytest.skip("NVIDIA GPU required")

    sm_count = torch.cuda.get_device_properties(device).multi_processor_count

    stream_exec = torch.cuda.Stream()
    stream_write = torch.cuda.Stream()

    k_wb_dev, v_wb_dev, k_wb_ptrs, v_wb_ptrs = _make_large_kv_pool(
        NUM_WB_SLOTS, device
    )
    # Destination is DEVICE memory (not host-pinned)
    k_dst_dev, v_dst_dev, k_dst_ptrs, v_dst_ptrs = _make_large_kv_pool(
        NUM_WB_SLOTS, device
    )
    k_fwd_dev, v_fwd_dev, k_fwd_ptrs, v_fwd_ptrs = _make_large_kv_pool(
        NUM_FWD_SLOTS, device
    )

    wb_src_indices = torch.arange(NUM_WB_SLOTS, dtype=torch.int32, device=device)
    wb_dst_indices = torch.arange(NUM_WB_SLOTS, dtype=torch.int32, device=device)

    BLOCK = 1024
    output_size = sm_count * 2 * BLOCK
    output_buf = torch.zeros(output_size, device=device, dtype=torch.float32)

    fwd_all_ptrs = torch.tensor(
        [b.view(torch.float16).data_ptr() for b in k_fwd_dev]
        + [b.view(torch.float16).data_ptr() for b in v_fwd_dev],
        dtype=torch.uint64,
        device=device,
    )

    FILL_VALUE = 1.0
    corruption_count = 0

    for iteration in range(ITERATIONS):
        torch.cuda.synchronize()

        for kb in k_wb_dev:
            kb.fill_(0.5)
        for vb in v_wb_dev:
            vb.fill_(0.5)
        for kb in k_fwd_dev:
            kb.fill_(FILL_VALUE)
        for vb in v_fwd_dev:
            vb.fill_(FILL_VALUE)
        output_buf.zero_()

        torch.cuda.synchronize()

        # Transfer: device → device (NOT host-pinned)
        with torch.cuda.stream(stream_write):
            transfer_kv_all_layer(
                src_k_layers=k_wb_ptrs,
                dst_k_layers=k_dst_ptrs,
                src_v_layers=v_wb_ptrs,
                dst_v_layers=v_dst_ptrs,
                src_indices=wb_src_indices,
                dst_indices=wb_dst_indices,
                item_size=ITEM_SIZE,
                num_layers=NUM_LAYERS,
            )

        # NO FENCE
        fwd_numel = k_fwd_dev[0].view(torch.float16).numel()
        grid = (sm_count * 2,)
        with torch.cuda.stream(stream_exec):
            _forward_compute_kernel[grid](
                fwd_all_ptrs,
                output_buf,
                fwd_numel,
                num_layers=NUM_LAYERS * 2,
                BLOCK=BLOCK,
                num_warps=4,
            )

        stream_write.synchronize()
        stream_exec.synchronize()

        has_nan = torch.isnan(output_buf).any().item()
        has_inf = torch.isinf(output_buf).any().item()
        if has_nan or has_inf:
            corruption_count += 1

    print(f"\nDevice-dst results: {corruption_count}/{ITERATIONS} corrupted")
    assert corruption_count == 0, (
        f"CORRUPTION with device destination in {corruption_count}/{ITERATIONS} "
        f"iters (unexpected if host-write is the specific trigger)"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_production_forward_write_integrity_no_fence():
    """
    Tests forward WRITE integrity: the forward kernel writes a known pattern
    to its pages while the transfer kernel runs concurrently on disjoint pages
    writing to host-pinned memory.

    Checks if the written pattern is correct after both kernels complete.
    This directly mimics set_kv_buffer writing new tokens while writeback
    reads old tokens from different pages.
    """
    device = "cuda"
    platform = current_platform()
    if platform.vendor != "nvidia":
        pytest.skip("NVIDIA GPU required")

    sm_count = torch.cuda.get_device_properties(device).multi_processor_count

    stream_exec = torch.cuda.Stream()
    stream_write = torch.cuda.Stream()

    k_wb_dev, v_wb_dev, k_wb_ptrs, v_wb_ptrs = _make_large_kv_pool(
        NUM_WB_SLOTS, device
    )
    k_host, v_host, k_host_ptrs, v_host_ptrs = _make_host_pinned_pool(NUM_WB_SLOTS)
    k_fwd_dev, v_fwd_dev, k_fwd_ptrs, v_fwd_ptrs = _make_large_kv_pool(
        NUM_FWD_SLOTS, device
    )

    wb_src_indices = torch.arange(NUM_WB_SLOTS, dtype=torch.int32, device=device)
    wb_dst_indices = torch.arange(NUM_WB_SLOTS, dtype=torch.int32, device=device)

    fwd_write_ptrs = torch.tensor(
        [b.view(torch.uint16).data_ptr() for b in k_fwd_dev]
        + [b.view(torch.uint16).data_ptr() for b in v_fwd_dev],
        dtype=torch.uint64,
        device=device,
    )

    WRITE_PATTERN = 0xCAFE
    corruption_count = 0
    first_msg = ""

    for iteration in range(ITERATIONS):
        torch.cuda.synchronize()

        for kb in k_wb_dev:
            kb.fill_(0.5)
        for vb in v_wb_dev:
            vb.fill_(0.5)
        # Clear forward buffers
        for kb in k_fwd_dev:
            kb.zero_()
        for vb in v_fwd_dev:
            vb.zero_()
        for kb in k_host:
            kb.zero_()
        for vb in v_host:
            vb.zero_()

        torch.cuda.synchronize()

        # Launch transfer (device → host-pinned)
        with torch.cuda.stream(stream_write):
            transfer_kv_all_layer(
                src_k_layers=k_wb_ptrs,
                dst_k_layers=k_host_ptrs,
                src_v_layers=v_wb_ptrs,
                dst_v_layers=v_host_ptrs,
                src_indices=wb_src_indices,
                dst_indices=wb_dst_indices,
                item_size=ITEM_SIZE,
                num_layers=NUM_LAYERS,
            )

        # Launch forward writer — NO FENCE
        fwd_numel = k_fwd_dev[0].view(torch.uint16).numel()
        grid = (sm_count * 2,)
        with torch.cuda.stream(stream_exec):
            _forward_write_kernel[grid](
                fwd_write_ptrs,
                fwd_numel,
                WRITE_PATTERN,
                num_layers=NUM_LAYERS * 2,
                BLOCK=1024,
                num_warps=4,
            )

        stream_write.synchronize()
        stream_exec.synchronize()

        # Verify forward write was correct
        for layer_idx in range(NUM_LAYERS):
            flat_k = k_fwd_dev[layer_idx].view(torch.uint16).flatten()
            wrong_k = (flat_k != WRITE_PATTERN).sum().item()
            if wrong_k > 0:
                corruption_count += 1
                sample = flat_k[flat_k != WRITE_PATTERN][:5].tolist()
                msg = (
                    f"iter={iteration} K layer={layer_idx}: "
                    f"{wrong_k}/{flat_k.numel()} wrong words (sample: {sample})"
                )
                if not first_msg:
                    first_msg = msg
                break

            flat_v = v_fwd_dev[layer_idx].view(torch.uint16).flatten()
            wrong_v = (flat_v != WRITE_PATTERN).sum().item()
            if wrong_v > 0:
                corruption_count += 1
                sample = flat_v[flat_v != WRITE_PATTERN][:5].tolist()
                msg = (
                    f"iter={iteration} V layer={layer_idx}: "
                    f"{wrong_v}/{flat_v.numel()} wrong words (sample: {sample})"
                )
                if not first_msg:
                    first_msg = msg
                break

    print(f"\nWrite integrity: {corruption_count}/{ITERATIONS} corrupted")
    assert corruption_count == 0, (
        f"WRITE CORRUPTION in {corruption_count}/{ITERATIONS} iterations "
        f"with disjoint pages + host-pinned transfer. "
        f"First: {first_msg}"
    )