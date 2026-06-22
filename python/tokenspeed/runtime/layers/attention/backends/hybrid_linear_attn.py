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

"""Hybrid linear attention backend for Qwen3.5 GDN models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
from tokenspeed_kernel.ops.attention.flashinfer import (
    gated_delta_rule as gdn_flashinfer,
)
from tokenspeed_kernel.ops.attention.triton.gdn_qkv_split import (
    fused_qkv_split_gdn_prefill,
)

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.linear.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from tokenspeed.runtime.layers.attention.linear.chunk_delta_h import (
    CHUNK_SIZE as FLA_CHUNK_SIZE,
)
from tokenspeed.runtime.layers.attention.linear.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from tokenspeed.runtime.layers.attention.linear.gdn import fused_gdn_gating
from tokenspeed.runtime.layers.attention.linear.index import (
    set_total_chunks_hint,
    set_total_chunks_hint_uniform,
)
from tokenspeed.runtime.layers.attention.linear.l2norm import l2norm_fwd
from tokenspeed.runtime.layers.attention.linear.mamba_state_scatter_triton import (
    fused_mamba_state_copy,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.configs.base import BaseAttnConfig
    from tokenspeed.runtime.layers.attention.kv_cache.base import BaseTokenToKVPool
    from tokenspeed.runtime.layers.paged_attention import PagedAttention


@dataclass
class MambaForwardMetadata:
    query_start_loc: torch.Tensor | None
    mamba_cache_indices: torch.Tensor
    mamba_output_indices: Optional[torch.Tensor] = None
    mamba_req_pool_indices: Optional[torch.Tensor] = None
    extend_prefix_lens: Optional[torch.Tensor] = None
    extend_seq_lens_cpu: Optional[torch.Tensor] = None
    # Pre-computed src/dst indices for extracting Mamba prefix-cache snapshots.
    track_ssm_h_src: Optional[torch.Tensor] = None
    track_ssm_h_dst: Optional[torch.Tensor] = None
    track_conv_indices: Optional[torch.Tensor] = None
    track_ssm_final_src: Optional[torch.Tensor] = None
    track_ssm_final_dst: Optional[torch.Tensor] = None
    # Triton fallback fields: batch sequence index and local token step for
    # each tracked request.  Used by fused_sigmoid_gating_delta_rule_update
    # via output_state_indices on platforms without sm100/sm103 (e.g. sm120).
    track_ssm_h_batch_indices: Optional[torch.Tensor] = None
    track_ssm_h_local_steps: Optional[torch.Tensor] = None


class LayerMappedKVPool:
    """Wraps a KV pool to map global layer IDs to internal pool indices.

    For hybrid models, only full attention layers have KV cache. This wrapper
    translates global layer IDs (e.g., 3, 7, 11) to pool indices (0, 1, 2).
    """

    def __init__(
        self, inner_pool: BaseTokenToKVPool, full_attention_layer_ids: list[int]
    ):
        self.inner = inner_pool
        self.layer_ids = list(full_attention_layer_ids)
        self.layer_map = {
            global_id: pool_idx
            for pool_idx, global_id in enumerate(full_attention_layer_ids)
        }
        # Expose page_size from inner pool for the scheduler
        self.page_size = getattr(inner_pool, "page_size", 1)

    def _map(self, layer_id: int) -> int:
        return self.layer_map.get(layer_id, layer_id)

    def set_kv_buffer(
        self,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor | None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
    ):
        orig = layer.layer_id
        layer.layer_id = self._map(orig)
        self.inner.set_kv_buffer(layer, out_cache_loc, k, v, k_scale, v_scale)
        layer.layer_id = orig

    def get_kv_buffer(self, layer_id: int):
        return self.inner.get_kv_buffer(self._map(layer_id))

    def get_key_buffer(self, layer_id: int):
        return self.inner.get_key_buffer(self._map(layer_id))

    def get_value_buffer(self, layer_id: int):
        return self.inner.get_value_buffer(self._map(layer_id))

    def __getattr__(self, name):
        return getattr(self.inner, name)


class SimpleMambaPool:
    """Mamba state pool indexed by scheduler-assigned cache slots.

    When *mamba_arena* is provided (a ``ChunkArena`` instance backed by a
    CUDA-VMM virtual-address window), ``conv_state`` and ``ssm_state`` are
    constructed as zero-copy views onto that arena instead of being allocated
    by the default CUDA allocator.  This enables HiMA Phase-2 inter-pool
    capacity transfers: growing/shrinking the arena maps/unmaps physical pages
    while the tensor VA remains stable, keeping captured CUDA graphs valid.

    Arena-backing is opt-in: if *mamba_arena* is ``None`` the pool allocates
    tensors normally via ``torch.zeros``, which is the default behaviour.
    """

    def __init__(
        self,
        size: int,
        num_mamba_layers: int,
        conv_state_shape: tuple,
        temporal_state_shape: tuple,
        conv_dtype: torch.dtype,
        ssm_dtype: torch.dtype,
        mamba_layer_ids: list[int],
        device: str,
        page_size: int = 1,
        speculative_num_draft_tokens: int = 0,
        max_req_pool_size: int = 0,
        mamba_arena: "object | None" = None,
    ):
        self.size = size
        self.device = device
        self.mamba_layer_ids = list(mamba_layer_ids)
        self.page_size = page_size
        self.mamba_map = {layer_id: i for i, layer_id in enumerate(mamba_layer_ids)}
        self.is_kda_cache = False
        self.max_req_pool_size = max_req_pool_size

        # Base slots (working + checkpoint) are allocated by C++ scheduler.
        # Python-only draft rows live after the scheduler-owned range and are
        # addressed by normal row indices in the same tensors.
        self.base_size = size
        self.speculative_num_draft_tokens = speculative_num_draft_tokens
        self.current_input_size = (
            max_req_pool_size + 1 if max_req_pool_size > 0 else size
        )
        self.draft_slots_per_req = max(0, speculative_num_draft_tokens - 1)
        self.draft_base = size
        self.draft_total_slots = self.current_input_size * self.draft_slots_per_req
        total_size = size + self.draft_total_slots
        self.total_size = total_size

        if mamba_arena is not None:
            self.conv_state, self.ssm_state = self._alloc_from_arena(
                mamba_arena=mamba_arena,
                num_mamba_layers=num_mamba_layers,
                total_size=total_size,
                conv_state_shape=conv_state_shape,
                temporal_state_shape=temporal_state_shape,
                conv_dtype=conv_dtype,
                ssm_dtype=ssm_dtype,
                device=device,
            )
        else:
            # Allocate conv state: (num_mamba_layers, total_size, conv_dim, state_len)
            self.conv_state = torch.zeros(
                num_mamba_layers,
                total_size,
                *conv_state_shape,
                dtype=conv_dtype,
                device=device,
            )
            # Allocate temporal/SSM state: (num_mamba_layers, total_size, heads, key_dim, val_dim)
            self.ssm_state = torch.zeros(
                num_mamba_layers,
                total_size,
                *temporal_state_shape,
                dtype=ssm_dtype,
                device=device,
            )

        self.mamba_cache = (self.conv_state, self.ssm_state)
        self.layer_transfer_counter = None
        self._mamba_arena: "object | None" = None  # set by bind_arena()

        self.current_input_indices = torch.full(
            (self.current_input_size,), -1, dtype=torch.int32, device=device
        )

    def bind_arena(self, mamba_arena: "object") -> None:
        """Rebind conv_state / ssm_state to a VMM ChunkArena VA window.

        This is the *post-construction* path used when the arena is created
        after the pool (e.g. in EventLoop._init_xpool_actuator).  The existing
        tensors are replaced with zero-copy views over the arena VA.  Any
        in-flight GPU work MUST be synchronised before calling this method.

        The arena must already be fully mapped (all chunks present) and sized
        to hold both conv_state and ssm_state.
        """
        import torch.cuda

        num_layers = self.conv_state.shape[0]
        conv_shape = tuple(self.conv_state.shape)
        ssm_shape = tuple(self.ssm_state.shape)
        conv_dtype = self.conv_state.dtype
        ssm_dtype = self.ssm_state.dtype

        new_conv, new_ssm = self._alloc_from_arena(
            mamba_arena=mamba_arena,
            num_mamba_layers=num_layers,
            total_size=self.total_size,
            conv_state_shape=conv_shape[2:],
            temporal_state_shape=ssm_shape[2:],
            conv_dtype=conv_dtype,
            ssm_dtype=ssm_dtype,
            device=self.device,
        )

        # Copy existing state into the new arena-backed tensors so that any
        # cached prefixes / checkpointed mamba states are preserved.
        with torch.no_grad():
            new_conv.copy_(self.conv_state)
            new_ssm.copy_(self.ssm_state)

        torch.cuda.synchronize()

        self.conv_state = new_conv
        self.ssm_state = new_ssm
        self.mamba_cache = (self.conv_state, self.ssm_state)
        self._mamba_arena = mamba_arena

    @staticmethod
    def _alloc_from_arena(
        *,
        mamba_arena: "object",
        num_mamba_layers: int,
        total_size: int,
        conv_state_shape: tuple,
        temporal_state_shape: tuple,
        conv_dtype: torch.dtype,
        ssm_dtype: torch.dtype,
        device: str,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """Build conv_state and ssm_state as views over a VMM arena VA window.

        Layout in the arena (contiguous, in order):
            [conv block: num_layers × total_size × *conv_shape  (conv_dtype)]
            [ssm  block: num_layers × total_size × *ssm_shape   (ssm_dtype)]

        The arena must already be fully mapped (all chunks present) when this
        is called.  After construction, capacity changes happen by growing /
        shrinking the arena WITHOUT changing the base pointer — so the tensor
        objects remain valid even inside CUDA graphs.
        """
        import math

        from tokenspeed.runtime.cache.arena.from_blob import from_blob

        conv_elem = torch.empty(0, dtype=conv_dtype).element_size()
        ssm_elem = torch.empty(0, dtype=ssm_dtype).element_size()

        conv_shape = (num_mamba_layers, total_size, *conv_state_shape)
        ssm_shape = (num_mamba_layers, total_size, *temporal_state_shape)

        conv_bytes = math.prod(conv_shape) * conv_elem
        ssm_bytes = math.prod(ssm_shape) * ssm_elem

        # Verify the arena has enough VA backing.
        arena_bytes = mamba_arena.max_chunks * mamba_arena.chunk_size
        needed = conv_bytes + ssm_bytes
        if needed > arena_bytes:
            raise RuntimeError(
                f"SimpleMambaPool: mamba_arena too small "
                f"({arena_bytes} bytes) for conv+ssm tensors ({needed} bytes). "
                f"Increase arena size or disable VMM pool binding."
            )

        base = mamba_arena.base_ptr
        conv_state = from_blob(base, conv_shape, dtype=conv_dtype, device=device)
        ssm_state = from_blob(
            base + conv_bytes, ssm_shape, dtype=ssm_dtype, device=device
        )
        # Zero-initialize (equivalent to torch.zeros).
        conv_state.zero_()
        ssm_state.zero_()
        return conv_state, ssm_state

    def get_mamba_indices(self, mamba_pool_indices: torch.Tensor) -> torch.Tensor:
        """Return mamba cache indices directly (allocated by C++ scheduler)."""
        return mamba_pool_indices.to(torch.int32)

    @staticmethod
    @torch.compile(dynamic=True)
    def _build_mtp_output_indices_kernel(
        output_indices: torch.Tensor,
        req_pool_indices: torch.Tensor,
        working_indices: torch.Tensor,
        draft_base: int,
        draft_slots_per_req: int,
        draft_token_num: int,
    ) -> None:
        """Fused fill of MTP target-verify output index table.

        Inductor fuses the working-column write and the draft-grid write into
        as few elementwise kernels as possible.  The host-side early returns
        (draft_token_num<=0, ``out is None``) are kept in the wrapper.
        """
        bs = working_indices.shape[0]
        working = working_indices.to(torch.int32)
        valid = working >= 0
        output_indices[:, 0] = torch.where(valid, working, -1)

        if draft_token_num > 1 and draft_slots_per_req > 0:
            req = req_pool_indices[:bs].to(torch.int32)
            steps = torch.arange(
                draft_token_num - 1, dtype=torch.int32, device=working.device
            )
            draft = draft_base + req[:, None] * draft_slots_per_req + steps[None, :]
            output_indices[:, 1:] = torch.where(
                valid[:, None] & (req >= 0)[:, None],
                draft,
                -1,
            )

    def get_mtp_output_indices(
        self,
        req_pool_indices: torch.Tensor,
        working_indices: torch.Tensor,
        draft_token_num: int,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build per-request target-verify outputs: [working, draft0, ...]."""
        bs = working_indices.shape[0]
        if out is not None:
            output_indices = out
            output_indices.fill_(-1)
        else:
            output_indices = torch.full(
                (bs, draft_token_num),
                -1,
                dtype=torch.int32,
                device=working_indices.device,
            )
        if draft_token_num <= 0:
            return output_indices

        self._build_mtp_output_indices_kernel(
            output_indices,
            req_pool_indices,
            working_indices,
            self.draft_base,
            self.draft_slots_per_req,
            draft_token_num,
        )
        return output_indices

    @staticmethod
    @torch.compile(dynamic=True)
    def _get_current_input_indices_kernel(
        req_pool_indices: torch.Tensor,
        working_indices: torch.Tensor,
        current_input_indices_buf: torch.Tensor,
        current_input_size: int,
    ) -> torch.Tensor:
        """Fused gather + masked-where for the no-COW path."""
        n = working_indices.shape[0]
        req = req_pool_indices[:n].to(torch.int32)
        working = working_indices.to(torch.int32)
        valid = (working >= 0) & (req >= 0)
        safe = req.clamp(0, current_input_size - 1).to(torch.int64)
        stored = current_input_indices_buf[safe]
        current = torch.where(valid & (stored >= 0), stored, working)
        current = torch.where(valid, current, torch.full_like(current, -1))
        return current

    @staticmethod
    @torch.compile(dynamic=True)
    def _get_current_input_indices_with_cow_kernel(
        req_pool_indices: torch.Tensor,
        working_indices: torch.Tensor,
        cow_src_indices: torch.Tensor,
        current_input_indices_buf: torch.Tensor,
        current_input_size: int,
    ) -> torch.Tensor:
        """Fused gather + masked-where for the COW path."""
        n = working_indices.shape[0]
        req = req_pool_indices[:n].to(torch.int32)
        working = working_indices.to(torch.int32)
        cow = cow_src_indices[:n].to(torch.int32)
        valid = (working >= 0) & (req >= 0)
        safe = req.clamp(0, current_input_size - 1).to(torch.int64)
        stored = current_input_indices_buf[safe]
        current = torch.where(valid & (stored >= 0), stored, working)
        current = torch.where(valid, current, torch.full_like(current, -1))
        current = torch.where(
            (cow >= 0) & valid & (current == working),
            cow,
            current,
        )
        return current

    def get_current_input_indices(
        self,
        req_pool_indices: torch.Tensor,
        working_indices: torch.Tensor,
        cow_src_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the row each request should read at the start of target verify."""
        if cow_src_indices is None:
            return self._get_current_input_indices_kernel(
                req_pool_indices,
                working_indices,
                self.current_input_indices,
                self.current_input_size,
            )
        return self._get_current_input_indices_with_cow_kernel(
            req_pool_indices,
            working_indices,
            cow_src_indices,
            self.current_input_indices,
            self.current_input_size,
        )

    def reset_current_inputs(
        self, req_pool_indices: torch.Tensor, working_indices: torch.Tensor
    ) -> None:
        """Mark freshly allocated/reused scheduler slots as canonical."""
        req_pool_indices = req_pool_indices[: working_indices.shape[0]].to(torch.int32)
        working_indices = working_indices.to(torch.int32)
        self.current_input_indices[req_pool_indices.long()] = working_indices

    @staticmethod
    @torch.compile(dynamic=True)
    def _update_current_inputs_after_verify_kernel(
        req_pool_indices: torch.Tensor,
        output_indices: torch.Tensor,
        accepted_lengths: torch.Tensor,
        current_input_indices: torch.Tensor,
        max_col: int,
    ) -> None:
        """Fused gather-scatter for the after-verify input pointer update.

        Inductor fuses clamp/arange/sub/dtype-convert into a single elementwise
        kernel; the gather and the in-place scatter on ``current_input_indices``
        each remain a single kernel.  All tensors stay on GPU; no host sync.
        """
        n = accepted_lengths.shape[0]
        req = req_pool_indices[:n].to(torch.int64)
        idx = (accepted_lengths.clamp(min=1, max=max_col) - 1).to(torch.int64)
        rows = torch.arange(n, device=accepted_lengths.device, dtype=torch.int64)
        selected = output_indices[rows, idx].to(torch.int32)
        current_input_indices[req] = selected

    def update_current_inputs_after_verify(
        self,
        req_pool_indices: torch.Tensor,
        output_indices: torch.Tensor,
        accepted_lengths: torch.Tensor,
    ) -> None:
        if output_indices is None or output_indices.numel() == 0:
            return
        self._update_current_inputs_after_verify_kernel(
            req_pool_indices,
            output_indices,
            accepted_lengths,
            self.current_input_indices,
            output_indices.shape[1],
        )

    def register_layer_transfer_counter(self, layer_transfer_counter):
        self.layer_transfer_counter = layer_transfer_counter

    def get_mamba_params(self, layer_id: int):
        """Return per-layer cache slices."""
        internal_idx = self.mamba_map[layer_id]
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(internal_idx)
        return [self.mamba_cache[i][internal_idx] for i in range(len(self.mamba_cache))]

    def get_mamba_params_all_layers(self):
        """Return all layers for all cache components."""
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(self.conv_state.shape[0] - 1)
        return [self.mamba_cache[i] for i in range(len(self.mamba_cache))]

    def get_contiguous_buf_infos(self):
        """Return per-layer mamba cache buffers for disaggregated transfer."""
        data_ptrs = []
        data_lens = []
        item_lens = []
        for cache in self.mamba_cache:
            for layer_id in range(cache.shape[0]):
                layer_cache = cache[layer_id]
                data_ptrs.append(layer_cache.data_ptr())
                data_lens.append(layer_cache.nbytes)
                item_lens.append(layer_cache[0].nbytes)
        return data_ptrs, data_lens, item_lens

    def get_contiguous_buf_unit_lens(self):
        unit_lens = []
        for cache in self.mamba_cache:
            for layer_id in range(cache.shape[0]):
                layer_cache = cache[layer_id]
                unit_lens.append(layer_cache[0, 0].nbytes)
        return unit_lens

    def get_contiguous_buf_layer_ids(self):
        """Return global layer ids aligned with get_contiguous_buf_infos()."""
        return self.mamba_layer_ids * len(self.mamba_cache)


class MambaAttnBackend(AttentionBackend):
    """Attention backend for Mamba/GDN linear attention layers."""

    def __init__(self, config: BaseAttnConfig):
        super().__init__(config)
        self.pad_slot_id = -1
        self.forward_metadata: MambaForwardMetadata = None
        self.state_indices_list = []
        self.query_start_loc_list = []
        self.cached_cuda_graph_decode_query_start_loc: torch.Tensor = None
        self.cached_cuda_graph_verify_query_start_loc: torch.Tensor = None
        self.output_indices_list = []
        self.speculative_num_draft_tokens = getattr(
            config, "speculative_num_draft_tokens", 0
        )
        self.pool: SimpleMambaPool = None
        self._gdn_fastpath_checked = False
        self._gdn_fastpath_avail = False

    def set_pool(self, pool: SimpleMambaPool):
        self.pool = pool

    def reset_current_inputs(
        self, req_pool_indices: torch.Tensor, working_indices: torch.Tensor
    ):
        if self.pool is not None:
            self.pool.reset_current_inputs(req_pool_indices, working_indices)

    def init_forward_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode = ForwardMode.DECODE,
        **kwargs,
    ):
        mamba_pool_indices = kwargs.get("mamba_pool_indices")
        if mamba_pool_indices is not None:
            mamba_cache_indices = self.pool.get_mamba_indices(mamba_pool_indices[:bs])
        else:
            mamba_cache_indices = self.pool.get_mamba_indices(req_pool_indices[:bs])

        is_target_verify = (
            forward_mode.is_decode_or_idle()
            and not self.is_draft
            and self.spec_num_tokens > 1
        )
        is_draft_extend = (
            forward_mode.is_decode_or_idle()
            and self.is_draft
            and self.spec_num_tokens > 1
        )

        mamba_output_indices = None
        extend_seq_lens_cpu = None
        if is_target_verify:
            draft_token_num = int(
                kwargs.get("tokens_per_req", self.speculative_num_draft_tokens)
            )
            cow_src_indices = kwargs.get("mamba_cow_src_indices")
            mamba_input_indices = self.pool.get_current_input_indices(
                req_pool_indices[:bs], mamba_cache_indices, cow_src_indices
            )
            mamba_output_indices = self.pool.get_mtp_output_indices(
                req_pool_indices[:bs],
                mamba_cache_indices,
                draft_token_num,
            )
            mamba_cache_indices = mamba_input_indices

        if forward_mode.is_decode_or_idle() and self.spec_num_tokens == 1:
            query_start_loc = torch.arange(
                0, bs + 1, dtype=torch.int32, device=self.device
            )
        elif forward_mode.is_extend_or_mixed() or is_target_verify or is_draft_extend:
            if is_target_verify or is_draft_extend:
                tokens_per_req = kwargs.get(
                    "tokens_per_req", self.speculative_num_draft_tokens
                )
                query_start_loc = torch.arange(
                    0,
                    bs * tokens_per_req + 1,
                    step=tokens_per_req,
                    dtype=torch.int32,
                    device=self.device,
                )
                set_total_chunks_hint_uniform(bs, tokens_per_req, query_start_loc)
            else:
                extend_start_loc = kwargs.get("extend_start_loc")
                extend_seq_lens = kwargs.get("extend_seq_lens")
                if extend_start_loc is not None and extend_seq_lens is not None:
                    query_start_loc = torch.empty(
                        (bs + 1,), dtype=torch.int32, device=self.device
                    )
                    query_start_loc[:bs] = extend_start_loc
                    query_start_loc[bs] = extend_start_loc[-1] + extend_seq_lens[-1]
                    extend_seq_lens_cpu = extend_seq_lens[:bs].to(
                        device="cpu", dtype=torch.int32
                    )
                else:
                    extend_prefix_lens = kwargs.get("extend_prefix_lens")
                    if extend_prefix_lens is not None:
                        extend_lens = (seq_lens[:bs] - extend_prefix_lens[:bs]).to(
                            torch.int32
                        )
                    else:
                        # No prefix: all tokens are new
                        extend_lens = seq_lens[:bs].to(torch.int32)
                    query_start_loc = torch.zeros(
                        bs + 1, dtype=torch.int32, device=self.device
                    )
                    torch.cumsum(extend_lens, dim=0, out=query_start_loc[1:])
                    extend_seq_lens_cpu = extend_lens.to(device="cpu")
                set_total_chunks_hint(extend_seq_lens_cpu, query_start_loc)
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")

        track_ssm_h_src = None
        track_ssm_h_dst = None
        track_conv_indices = None
        track_ssm_final_src = None
        track_ssm_final_dst = None
        track_ssm_h_batch_indices = None
        track_ssm_h_local_steps = None
        if (
            forward_mode.is_extend_or_mixed() or is_draft_extend
        ) and not is_target_verify:
            extend_prefix_lens_kw = kwargs.get("extend_prefix_lens")
            mamba_track_pool_indices = kwargs.get("mamba_track_pool_indices")
            if (
                extend_prefix_lens_kw is not None
                and mamba_track_pool_indices is not None
            ):
                prefix = extend_prefix_lens_kw[:bs].to(
                    dtype=torch.int32, device=self.device
                )
                track_indices = mamba_track_pool_indices[:bs].to(
                    dtype=torch.int32, device=self.device
                )
                extend_lens = (seq_lens[:bs] - prefix).to(torch.int32)
                checkpoint_mask = (track_indices >= 0) & (mamba_cache_indices >= 0)

                page_size = getattr(self.pool, "page_size", 1)
                final_lens = prefix + extend_lens
                last_inserted_lens = (final_lens // page_size) * page_size
                track_lens = last_inserted_lens - prefix
                track_inside = (
                    checkpoint_mask & (track_lens > 0) & (track_lens < extend_lens)
                )
                track_mask = track_inside & ((track_lens % FLA_CHUNK_SIZE) == 0)
                # C++ attaches the checkpoint slot to the last KV page inserted
                # for this chunk. When a chunk has an intermediate branch and
                # ends exactly on a page boundary, the final state must win.
                final_mask = (
                    checkpoint_mask
                    & (final_lens >= page_size)
                    & ((final_lens % page_size) == 0)
                )
                if final_mask.any():
                    track_ssm_final_src = mamba_cache_indices[final_mask]
                    track_ssm_final_dst = track_indices[final_mask]

                if track_mask.any():
                    (
                        track_ssm_h_src,
                        track_ssm_h_dst,
                        track_ssm_h_batch_indices,
                        track_ssm_h_local_steps,
                    ) = self._compute_track_ssm_indices(
                        track_lens,
                        track_mask,
                        track_indices,
                        seq_lens[:bs] - prefix,  # extend_seq_lens
                    )
                    track_conv_indices = self._compute_track_conv_indices(
                        query_start_loc,
                        track_lens,
                        track_mask,
                    )

        self.forward_metadata = MambaForwardMetadata(
            query_start_loc=query_start_loc,
            mamba_cache_indices=mamba_cache_indices,
            mamba_output_indices=mamba_output_indices,
            mamba_req_pool_indices=req_pool_indices[:bs],
            extend_prefix_lens=kwargs.get("extend_prefix_lens"),
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            track_ssm_h_src=track_ssm_h_src,
            track_ssm_h_dst=track_ssm_h_dst,
            track_conv_indices=track_conv_indices,
            track_ssm_final_src=track_ssm_final_src,
            track_ssm_final_dst=track_ssm_final_dst,
            track_ssm_h_batch_indices=track_ssm_h_batch_indices,
            track_ssm_h_local_steps=track_ssm_h_local_steps,
        )

    def _compute_track_conv_indices(
        self,
        query_start_loc: torch.Tensor,
        track_lens: torch.Tensor,
        track_mask: torch.Tensor,
    ):
        """Compute packed input indices for conv windows at tracked boundaries."""
        conv_state_len = self.pool.conv_state.shape[-1]
        lens_m = track_lens[track_mask]
        start = query_start_loc[:-1][track_mask] + lens_m - conv_state_len
        indices = start.unsqueeze(-1) + torch.arange(
            conv_state_len,
            device=self.device,
            dtype=start.dtype,
        )
        return indices.clamp(0, query_start_loc[-1] - 1)

    def _compute_track_ssm_indices(
        self,
        track_lens: torch.Tensor,
        track_mask: torch.Tensor,
        mamba_track_indices: torch.Tensor,
        extend_seq_lens: torch.Tensor,
    ):
        """Compute src/dst indices for extracting intermediate SSM states.

        Matching conv windows are gathered separately from packed pre-conv inputs.

        Returns:
            track_ssm_h_src: global flashinfer checkpoint indices (fast-path).
            track_ssm_h_dst: destination slots in ssm_states.
            batch_indices: per-tracked-seq batch index (Triton fallback).
            local_steps: per-tracked-seq local step at chunk boundary
                         (0-based within that sequence, Triton fallback).
        """
        # flashinfer ckpts[k] = state after processing chunk k = FLA h[k+1]
        num_fi_ckpts = extend_seq_lens // FLA_CHUNK_SIZE
        offset = torch.zeros_like(num_fi_ckpts)
        offset[1:] = torch.cumsum(num_fi_ckpts[:-1], dim=0)

        lens_m = track_lens[track_mask]
        offset_m = offset[track_mask]
        dst_m = mamba_track_indices[track_mask]

        # FLA h[lens//C] = flashinfer ckpts[lens//C - 1].
        # track_mask guarantees lens_m >= FLA_CHUNK_SIZE so lens_m // C >= 1.
        track_ssm_h_src = offset_m + (lens_m // FLA_CHUNK_SIZE - 1)
        track_ssm_h_dst = dst_m

        # Triton fallback: batch index (sequence number within the batch) and
        # local token step at the last complete chunk boundary (0-indexed).
        # The Triton kernel processes each sequence from local step 0, so the
        # state after chunk c is found at local step c*FLA_CHUNK_SIZE - 1.
        batch_indices = torch.where(track_mask)[0]
        local_steps = lens_m // FLA_CHUNK_SIZE * FLA_CHUNK_SIZE - 1

        return (
            track_ssm_h_src,
            track_ssm_h_dst,
            batch_indices,
            local_steps,
        )

    # ---- CUDA graph state ----

    def init_cuda_graph_state(
        self, max_num_tokens: int, seq_lens_buf: torch.Tensor = None
    ):
        del seq_lens_buf  # mamba doesn't use seq_lens_buf.
        for i in range(max_num_tokens):
            self.state_indices_list.append(
                torch.full(
                    (i + 1,), self.pad_slot_id, dtype=torch.int32, device=self.device
                )
            )
            self.query_start_loc_list.append(
                torch.empty((i + 2,), dtype=torch.int32, device=self.device)
            )
            if self.speculative_num_draft_tokens > 0:
                self.output_indices_list.append(
                    torch.full(
                        (i + 1, self.speculative_num_draft_tokens),
                        self.pad_slot_id,
                        dtype=torch.int32,
                        device=self.device,
                    )
                )
        self.cached_cuda_graph_decode_query_start_loc = torch.arange(
            0, max_num_tokens + 1, dtype=torch.int32, device=self.device
        )
        if self.speculative_num_draft_tokens > 0:
            # Need max_num_tokens+1 entries (one per request + sentinel).
            # Each entry is request_index * spec_num_draft_tokens.
            self.cached_cuda_graph_verify_query_start_loc = torch.arange(
                0,
                (max_num_tokens + 1) * self.speculative_num_draft_tokens,
                step=self.speculative_num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
        self._qsl_dirty = [False] * max_num_tokens
        self._qsl_last_mode = [None] * max_num_tokens

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        **kwargs,
    ):
        is_target_verify = (
            forward_mode.is_decode_or_idle()
            and not self.is_draft
            and self.spec_num_tokens > 1
        )
        is_draft_extend = (
            forward_mode.is_decode_or_idle()
            and self.is_draft
            and self.spec_num_tokens > 1
        )

        if forward_mode.is_decode_or_idle() and self.spec_num_tokens == 1:
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
            )
        elif is_target_verify or is_draft_extend:
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
            )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")

        mamba_pool_indices = kwargs.get("mamba_pool_indices")
        # Reuse the pre-allocated [bs]-length buffer as mamba_indices so the
        # capture path matches the replay path: zero allocation, single write.
        padded_mamba_indices = self.state_indices_list[bs - 1]
        if mamba_pool_indices is not None:
            padded_mamba_indices[:bs].copy_(
                self.pool.get_mamba_indices(mamba_pool_indices[:bs])
            )
        else:
            padded_mamba_indices[:bs].copy_(
                self.pool.get_mamba_indices(req_pool_indices[:bs])
            )
        mamba_output_indices = None
        if is_target_verify:
            cow_src_indices = kwargs.get("mamba_cow_src_indices")
            mamba_input_indices = self.pool.get_current_input_indices(
                req_pool_indices[:bs], padded_mamba_indices, cow_src_indices
            )
            mamba_output_indices = self.output_indices_list[bs - 1]
            self.pool.get_mtp_output_indices(
                req_pool_indices[:bs],
                padded_mamba_indices,
                self.speculative_num_draft_tokens,
                out=mamba_output_indices,
            )
            padded_mamba_indices.copy_(mamba_input_indices)
        self._qsl_dirty[bs - 1] = False
        self._qsl_last_mode[bs - 1] = (forward_mode, self.spec_num_tokens > 1)
        self.forward_metadata = MambaForwardMetadata(
            query_start_loc=self.query_start_loc_list[bs - 1],
            mamba_cache_indices=self.state_indices_list[bs - 1],
            mamba_output_indices=mamba_output_indices,
            mamba_req_pool_indices=req_pool_indices[:bs],
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode = None,
        req_to_page: torch.Tensor = None,
        **kwargs,
    ):
        num_padding = kwargs.get("num_padding", 0)
        mamba_pool_indices = kwargs.get("mamba_pool_indices")

        real_bs = bs - num_padding
        req_pool_indices = req_pool_indices[:bs]

        # Reuse the pre-allocated [bs]-length buffer as the padded mamba_indices
        # so downstream ops (get_mtp_output_indices, get_current_input_indices)
        # see the full-batch shape with padding rows already set to -1.
        # Zero extra allocations on this hot path.
        padded_mamba_indices = self.state_indices_list[bs - 1]
        if mamba_pool_indices is not None:
            padded_mamba_indices[:real_bs].copy_(
                self.pool.get_mamba_indices(mamba_pool_indices[:real_bs])
            )
        else:
            padded_mamba_indices[:real_bs].copy_(
                self.pool.get_mamba_indices(req_pool_indices[:real_bs])
            )
        if num_padding > 0:
            padded_mamba_indices[real_bs:].fill_(-1)

        is_target_verify = (
            forward_mode is not None
            and forward_mode.is_decode_or_idle()
            and not self.is_draft
            and self.spec_num_tokens > 1
        )
        is_draft_extend = (
            forward_mode is not None
            and forward_mode.is_decode_or_idle()
            and self.is_draft
            and self.spec_num_tokens > 1
        )

        mamba_output_indices = None
        if is_target_verify:
            cow_src_indices = kwargs.get("mamba_cow_src_indices")
            mamba_input_indices = self.pool.get_current_input_indices(
                req_pool_indices, padded_mamba_indices, cow_src_indices
            )
            mamba_output_indices = self.output_indices_list[bs - 1]
            self.pool.get_mtp_output_indices(
                req_pool_indices,
                padded_mamba_indices,
                self.speculative_num_draft_tokens,
                out=mamba_output_indices,
            )
            # mamba_input_indices already encodes padding via padded_mamba_indices.
            padded_mamba_indices.copy_(mamba_input_indices)

        if num_padding == 0:
            need_copy = self._qsl_dirty[bs - 1] or self._qsl_last_mode[bs - 1] != (
                forward_mode,
                self.spec_num_tokens > 1,
            )
            if need_copy:
                if forward_mode.is_decode_or_idle() and self.spec_num_tokens == 1:
                    self.query_start_loc_list[bs - 1].copy_(
                        self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
                    )
                elif is_target_verify or is_draft_extend:
                    self.query_start_loc_list[bs - 1].copy_(
                        self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
                    )
                self._qsl_dirty[bs - 1] = False
                self._qsl_last_mode[bs - 1] = (forward_mode, self.spec_num_tokens > 1)
        else:
            if forward_mode.is_decode_or_idle() and self.spec_num_tokens == 1:
                self.query_start_loc_list[bs - 1][:real_bs].copy_(
                    self.cached_cuda_graph_decode_query_start_loc[:real_bs]
                )
                self.query_start_loc_list[bs - 1][real_bs:].fill_(real_bs)
            elif is_target_verify or is_draft_extend:
                self.query_start_loc_list[bs - 1][:real_bs].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[:real_bs]
                )
                self.query_start_loc_list[bs - 1][real_bs:].fill_(
                    real_bs * self.speculative_num_draft_tokens
                )
            else:
                raise ValueError(f"Invalid forward mode: {forward_mode=}")
            self._qsl_dirty[bs - 1] = True
            self._qsl_last_mode[bs - 1] = (forward_mode, self.spec_num_tokens > 1)

        self.forward_metadata = MambaForwardMetadata(
            query_start_loc=self.query_start_loc_list[bs - 1],
            mamba_cache_indices=self.state_indices_list[bs - 1],
            mamba_output_indices=mamba_output_indices,
            mamba_req_pool_indices=req_pool_indices,
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    # ---- Forward ----

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        # Multi-token decode (target verify or drafter compound) reuses
        # the multi-token kernel path in forward_extend. `q` is None for
        # hybrid linear-attn layers; the token count comes from mixed_qkv.
        q_len_per_req = kwargs["mixed_qkv"].shape[0] // bs if bs > 0 else 1
        if q_len_per_req > 1:
            return self.forward_extend(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                bs,
                forward_mode=ForwardMode.DECODE,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )

        mixed_qkv = kwargs["mixed_qkv"]
        conv_weights = kwargs["conv_weights"]
        bias = kwargs["bias"]
        activation = kwargs["activation"]
        key_dim = kwargs["key_dim"]
        value_dim = kwargs["value_dim"]
        attn_tp_size = kwargs["attention_tp_size"]
        head_k_dim = kwargs["head_k_dim"]
        head_v_dim = kwargs["head_v_dim"]
        a = kwargs["a"]
        b = kwargs["b"]
        A_log = kwargs["A_log"]
        dt_bias = kwargs["dt_bias"]
        layer_id = kwargs["layer_id"]

        conv_states, ssm_states, *rest = self.pool.get_mamba_params(layer_id)
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            conv_weights,
            bias,
            activation,
            conv_state_indices=cache_indices,
        )

        query, key, value = torch.split(
            mixed_qkv,
            [
                key_dim // attn_tp_size,
                key_dim // attn_tp_size,
                value_dim // attn_tp_size,
            ],
            dim=-1,
        )
        seq_len = query.shape[0]
        num_heads = query.shape[1] // head_k_dim
        query = query.view(1, seq_len, num_heads, head_k_dim)
        key = key.view(1, seq_len, num_heads, head_k_dim)
        value = value.view(1, seq_len, value.shape[1] // head_v_dim, head_v_dim)

        core_attn_out = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
        )
        return core_attn_out

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        forward_mode: ForwardMode,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        mixed_qkv = kwargs["mixed_qkv"]
        conv_weights = kwargs["conv_weights"]
        bias = kwargs["bias"]
        activation = kwargs["activation"]
        key_dim = kwargs["key_dim"]
        value_dim = kwargs["value_dim"]
        attn_tp_size = kwargs["attention_tp_size"]
        head_k_dim = kwargs["head_k_dim"]
        head_v_dim = kwargs["head_v_dim"]
        a = kwargs["a"]
        b = kwargs["b"]
        A_log = kwargs["A_log"]
        dt_bias = kwargs["dt_bias"]
        layer_id = kwargs["layer_id"]
        seq_len = kwargs["seq_len"]

        # `q` is None for hybrid linear-attn layers; the token count comes
        # from seq_len carried in kwargs.
        q_len_per_req = seq_len // bs if bs > 0 else 1
        is_target_verify = (
            forward_mode is not None
            and forward_mode.is_decode_or_idle()
            and not self.is_draft
            and q_len_per_req > 1
        )

        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        if is_target_verify:
            draft_token_num = kwargs.get(
                "draft_token_num", self.speculative_num_draft_tokens
            )
            conv_states, ssm_states = self.pool.get_mamba_params(layer_id)
            output_indices = self.forward_metadata.mamba_output_indices

            batch_size = seq_len // draft_token_num
            # shouldn't use contiguous here, because causal_conv1d_update
            # support input non-contiguous
            mixed_qkv_reshaped = mixed_qkv.view(
                batch_size, draft_token_num, -1
            ).transpose(1, 2)
            mixed_qkv_processed = causal_conv1d_update(
                mixed_qkv_reshaped,
                conv_states,
                conv_weights,
                bias,
                activation,
                conv_state_indices=cache_indices[:batch_size],
                output_state_indices=output_indices[:batch_size],
            )
            # needn't contiguous here.
            mixed_qkv = mixed_qkv_processed.transpose(1, 2).view(seq_len, -1)
        else:
            conv_states, ssm_states = self.pool.get_mamba_params(layer_id)
            extend_prefix_lens = kwargs.get("extend_prefix_lens")
            if extend_prefix_lens is None:
                extend_prefix_lens = self.forward_metadata.extend_prefix_lens
            extend_seq_lens_cpu = kwargs.get("extend_seq_lens_cpu")
            if extend_seq_lens_cpu is None:
                extend_seq_lens_cpu = self.forward_metadata.extend_seq_lens_cpu
            has_initial_states = (
                extend_prefix_lens > 0 if extend_prefix_lens is not None else None
            )
            need_h_track = (
                self.forward_metadata.track_ssm_h_src is not None
                and self.forward_metadata.track_ssm_h_src.numel() > 0
            )

            mixed_qkv_t = mixed_qkv.transpose(0, 1)
            if need_h_track:
                if self.forward_metadata.track_conv_indices is None:
                    raise RuntimeError(
                        "Missing conv indices for intermediate mamba track"
                    )
                conv_states[self.forward_metadata.track_ssm_h_dst] = mixed_qkv_t[
                    :, self.forward_metadata.track_conv_indices
                ].transpose(0, 1)

            mixed_qkv = causal_conv1d_fn(
                mixed_qkv_t,
                conv_weights,
                bias,
                activation=activation,
                conv_states=conv_states,
                has_initial_state=has_initial_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=extend_seq_lens_cpu,
            ).transpose(0, 1)[:seq_len]

        key_split_dim = key_dim // attn_tp_size
        value_split_dim = value_dim // attn_tp_size
        num_heads = key_split_dim // head_k_dim
        num_value_heads = value_split_dim // head_v_dim

        query, key, value = fused_qkv_split_gdn_prefill(
            mixed_qkv,
            num_q_heads=num_heads,
            num_k_heads=num_heads,
            num_v_heads=num_value_heads,
            head_q=head_k_dim,
            head_k=head_k_dim,
            head_v=head_v_dim,
        )

        if is_target_verify:
            draft_token_num = kwargs.get(
                "draft_token_num", self.speculative_num_draft_tokens
            )
            core_attn_out = fused_sigmoid_gating_delta_rule_update(
                A_log=A_log,
                dt_bias=dt_bias,
                q=query,
                k=key,
                v=value,
                a=a,
                b=b,
                initial_state_source=ssm_states,
                initial_state_indices=cache_indices,
                cu_seqlens=query_start_loc,
                use_qk_l2norm_in_kernel=True,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                # target_verify specific parameters
                disable_state_update=True,
                output_state_indices=self.forward_metadata.mamba_output_indices,
            )
        else:
            beta = b.sigmoid()
            g = fused_gdn_gating(A_log, a, dt_bias)
            g = g.unsqueeze(0)
            beta = beta.unsqueeze(0)

            recurrent_state = ssm_states[cache_indices]
            need_final_track = (
                self.forward_metadata.track_ssm_final_src is not None
                and self.forward_metadata.track_ssm_final_src.numel() > 0
            )
            if not self._gdn_fastpath_checked:
                self._gdn_fastpath_avail = (
                    gdn_flashinfer.is_supported(
                        head_k_dim, query.dtype, num_heads, num_value_heads
                    )
                    and head_v_dim == head_k_dim
                )
                if not self._gdn_fastpath_avail:
                    import logging as _logging

                    _logging.getLogger(__name__).warning(
                        "GDN prefill: flashinfer sm100/sm103 fast-path unavailable "
                        "(dtype=%s, head_k=%d, head_v=%d, num_q=%d, num_v=%d, "
                        "sm100_available=%s). Falling back to Triton recurrent path "
                        "(correct but slower; requires sm100/sm103 for full speed).",
                        query.dtype,
                        head_k_dim,
                        head_v_dim,
                        num_heads,
                        num_value_heads,
                        gdn_flashinfer.is_available(),
                    )
                self._gdn_fastpath_checked = True
            if self._gdn_fastpath_avail:
                if need_h_track:
                    core_attn_out, last_recurrent_state, fi_h_checkpoints, _ = (
                        gdn_flashinfer.gdn_chunk_prefill(
                            l2norm_fwd(query),
                            l2norm_fwd(key),
                            value,
                            g,
                            beta,
                            scale=head_k_dim**-0.5,
                            initial_state=recurrent_state,
                            cu_seqlens=query_start_loc,
                            output_h=True,
                        )
                    )
                else:
                    core_attn_out, last_recurrent_state = (
                        gdn_flashinfer.gdn_chunk_prefill(
                            l2norm_fwd(query),
                            l2norm_fwd(key),
                            value,
                            g,
                            beta,
                            scale=head_k_dim**-0.5,
                            initial_state=recurrent_state,
                            cu_seqlens=query_start_loc,
                        )
                    )
                last_recurrent_state = last_recurrent_state.to(
                    ssm_states.dtype, copy=False
                )
                ssm_states[cache_indices] = last_recurrent_state
            else:
                # Triton recurrent fallback for platforms without sm100/sm103
                # (e.g. consumer Blackwell sm120).  Numerically equivalent to
                # the flashinfer fast-path but processes tokens sequentially.
                #
                # h-tracking (mamba prefix-cache state checkpointing) is
                # implemented here via output_state_indices: the Triton kernel
                # writes intermediate hidden states directly into ssm_states at
                # the specified destination slots, matching what the flashinfer
                # fi_h_checkpoints mechanism achieves on the fast path.
                out_state_indices = None
                if need_h_track:
                    # query is [1, total_T, H, K] (VARLEN packed format).
                    # output_state_indices shape: (N_seqs, total_T), indexed as
                    # [batch_idx * total_T + local_step] inside the kernel.
                    total_T = query.shape[1]
                    num_seqs = query_start_loc.shape[0] - 1
                    out_state_indices = torch.full(
                        (num_seqs, total_T),
                        -1,
                        dtype=torch.int32,
                        device=query.device,
                    )
                    b_idx = self.forward_metadata.track_ssm_h_batch_indices
                    l_step = self.forward_metadata.track_ssm_h_local_steps
                    dst_slots = self.forward_metadata.track_ssm_h_dst
                    out_state_indices[b_idx, l_step] = dst_slots.to(torch.int32)

                core_attn_out = fused_sigmoid_gating_delta_rule_update(
                    A_log=A_log,
                    a=a,
                    dt_bias=dt_bias,
                    softplus_beta=1.0,
                    softplus_threshold=20.0,
                    q=query,
                    k=key,
                    v=value,
                    b=b,
                    initial_state_source=ssm_states,
                    initial_state_indices=cache_indices,
                    scale=head_k_dim**-0.5,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=query_start_loc,
                    output_state_indices=out_state_indices,
                )
                # Final states written in-place; intermediate h-checkpoints
                # (if any) written via output_state_indices above.

            if need_h_track and self._gdn_fastpath_avail:
                # Fast path: copy flashinfer's fi_h_checkpoints to ssm_states.
                # Triton fallback handles this internally via output_state_indices.
                ssm_states[self.forward_metadata.track_ssm_h_dst] = fi_h_checkpoints[
                    self.forward_metadata.track_ssm_h_src
                ].to(ssm_states.dtype, copy=False)

            if need_final_track:
                fused_mamba_state_copy(
                    conv_states,
                    self.forward_metadata.track_ssm_final_src,
                    self.forward_metadata.track_ssm_final_dst,
                )
                fused_mamba_state_copy(
                    ssm_states,
                    self.forward_metadata.track_ssm_final_src,
                    self.forward_metadata.track_ssm_final_dst,
                )

        return core_attn_out


class HybridLinearAttnBackend(AttentionBackend):
    """Hybrid backend that routes between full attention and linear attention by layer ID."""

    def __init__(
        self,
        full_attn_backend: AttentionBackend,
        linear_attn_backend: MambaAttnBackend,
        full_attn_layers: list[int],
    ):
        self.device = full_attn_backend.device
        self.full_attn_layers = set(full_attn_layers)
        self.full_attn_backend = full_attn_backend
        self.linear_attn_backend = linear_attn_backend

    def _backends(self):
        return [self.full_attn_backend, self.linear_attn_backend]

    def _backend_for_layer(self, layer_id: int) -> AttentionBackend:
        if self.linear_attn_backend is None or layer_id in self.full_attn_layers:
            return self.full_attn_backend
        return self.linear_attn_backend

    _MAMBA_KWARGS = frozenset(
        {
            "mamba_pool_indices",
            "mamba_cow_src_indices",
            "mamba_branching_seqlens",
            "mamba_track_pool_indices",
            "mamba_cache_chunk_size",
        }
    )

    @staticmethod
    def _split_mamba_kwargs(kwargs: dict) -> tuple[dict, dict]:
        mamba_kw = {}
        common_kw = {}
        for k, v in kwargs.items():
            if k in HybridLinearAttnBackend._MAMBA_KWARGS:
                mamba_kw[k] = v
            else:
                common_kw[k] = v
        return common_kw, mamba_kw

    # ---- Metadata delegation ----

    def init_forward_metadata(self, *args, **kwargs):
        common_kw, mamba_kw = self._split_mamba_kwargs(kwargs)
        self.full_attn_backend.init_forward_metadata(*args, **common_kw)
        self.linear_attn_backend.init_forward_metadata(*args, **common_kw, **mamba_kw)

    def init_cuda_graph_state(self, max_bs: int, seq_lens_buf: torch.Tensor):
        for backend in self._backends():
            backend.init_cuda_graph_state(max_bs, seq_lens_buf)

    def register_step_counter(self, step_counter):
        # Hybrid layerwise transfer needs one global step per model layer,
        # including both full-attention and mamba layers. Record steps in this
        # wrapper instead of in child backends to avoid double counting.
        self.step_counter = step_counter

    def init_forward_metadata_capture_cuda_graph(self, *args, **kwargs):
        common_kw, mamba_kw = self._split_mamba_kwargs(kwargs)
        self.full_attn_backend.init_forward_metadata_capture_cuda_graph(
            *args, **common_kw
        )
        self.linear_attn_backend.init_forward_metadata_capture_cuda_graph(
            *args, **common_kw, **mamba_kw
        )

    def init_forward_metadata_replay_cuda_graph(self, *args, **kwargs):
        common_kw, mamba_kw = self._split_mamba_kwargs(kwargs)
        self.full_attn_backend.init_forward_metadata_replay_cuda_graph(
            *args, **common_kw
        )
        self.linear_attn_backend.init_forward_metadata_replay_cuda_graph(
            *args, **common_kw, **mamba_kw
        )

    def support_kv_cache_prewrite(
        self, forward_mode: ForwardMode | None = None
    ) -> bool:
        return self.full_attn_backend.support_kv_cache_prewrite(forward_mode)

    # ---- Forward dispatch ----

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc,
        token_to_kv_pool,
        forward_mode: ForwardMode,
        bs: int,
        save_kv_cache: bool = True,
        record_kv_cache: bool | None = None,
        **kwargs,
    ):
        if forward_mode is None:
            return super().forward(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                forward_mode,
                bs,
                save_kv_cache,
                record_kv_cache=record_kv_cache,
                **kwargs,
            )

        if forward_mode.is_idle():
            if layer is None:
                return torch.empty_like(kwargs["z"])
            return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)

        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        backend = self._backend_for_layer(layer_id)

        # See AttentionBackend.forward for the record_kv_cache contract; the step
        # is recorded in this wrapper (not the child backends) to keep one step
        # per model layer across full-attn + mamba. Idle already returned above.
        with self.record_pd_cache_step(forward_mode, save_kv_cache, record_kv_cache):
            if forward_mode.is_decode():
                ret = backend.forward_decode(
                    q,
                    k,
                    v,
                    layer,
                    out_cache_loc,
                    token_to_kv_pool,
                    bs,
                    save_kv_cache=save_kv_cache,
                    **kwargs,
                )
            else:
                ret = backend.forward_extend(
                    q,
                    k,
                    v,
                    layer,
                    out_cache_loc,
                    token_to_kv_pool,
                    bs,
                    save_kv_cache=save_kv_cache,
                    forward_mode=forward_mode,
                    **kwargs,
                )
        return ret

    def forward_decode(
        self, q, k, v, layer, out_cache_loc, token_to_kv_pool, bs, **kwargs
    ):
        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        return self._backend_for_layer(layer_id).forward_decode(
            q, k, v, layer, out_cache_loc, token_to_kv_pool, bs, **kwargs
        )

    def forward_extend(
        self, q, k, v, layer, out_cache_loc, token_to_kv_pool, bs, **kwargs
    ):
        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        return self._backend_for_layer(layer_id).forward_extend(
            q, k, v, layer, out_cache_loc, token_to_kv_pool, bs, **kwargs
        )

    def reset_current_inputs(self, *args, **kwargs):
        if self.linear_attn_backend is None:
            return
        if hasattr(self.linear_attn_backend, "reset_current_inputs"):
            self.linear_attn_backend.reset_current_inputs(*args, **kwargs)

    def update_mamba_state_after_mtp_verify(self, accepted_length, model):
        # mamba_cache_indices are input rows during target-verify. The first
        # output row is always the scheduler-owned working slot, so use the
        # output index table to update the next-round input pointer.
        output_indices = self.linear_attn_backend.forward_metadata.mamba_output_indices
        if output_indices is None:
            return
        req_pool_indices = (
            self.linear_attn_backend.forward_metadata.mamba_req_pool_indices
        )
        if req_pool_indices is None:
            return
        request_number = accepted_length.shape[0]
        self.linear_attn_backend.pool.update_current_inputs_after_verify(
            req_pool_indices[:request_number],
            output_indices[:request_number],
            accepted_length,
        )
