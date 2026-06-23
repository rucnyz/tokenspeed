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

import faulthandler
import json
import os
import signal
import time
from collections import OrderedDict
from dataclasses import dataclass

import psutil
import setproctitle
import torch
import torch.distributed as dist
import zmq
from tokenspeed_scheduler import PD, Cache, ExecutionEvent, Scheduler

from tokenspeed.runtime.cache.executor.memory_executor import (
    MemoryExecutor,
    MemoryExecutorConfig,
)
from tokenspeed.runtime.cache.transfer.types import CacheKind
from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.engine.generation_output_processor import OutputProcesser
from tokenspeed.runtime.engine.memory_occupation import MemoryOccupationController
from tokenspeed.runtime.engine.pause import PauseController
from tokenspeed.runtime.engine.request_handler import RequestHandler
from tokenspeed.runtime.engine.scheduler_utils import (
    advance_forward,
    cache_event_from_payload,
    cache_event_key,
    cache_event_to_payload,
    cache_sync_debug_enabled,
    compute_lpb_byte_sizes,
    make_config,
    pool_to_paged_cache_groups,
    pool_to_prefix_cache_adjunct_spec,
    pop_common_cache_event_payloads,
    should_use_overlap_schedule,
)
from tokenspeed.runtime.execution.distributed_initializer import (
    DistributedConfig,
    DistributedInitializer,
)
from tokenspeed.runtime.execution.factory import (
    ModelExecutorConfig,
    create_model_executor,
    create_model_runner,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.types import ModelExecutionResult
from tokenspeed.runtime.grammar.capturable_grammar import GrammarStepInputs
from tokenspeed.runtime.layers.attention.registry import create_attn_components
from tokenspeed.runtime.metrics.collector import EngineMetrics
from tokenspeed.runtime.pd.decode_executor import DisaggDecodeExecutor
from tokenspeed.runtime.pd.factory import (
    create_pd_kv_transfer,
    get_kv_args,
)
from tokenspeed.runtime.pd.kv_events import (
    EventPublisherFactory,
    KVEventBatch,
    NullEventPublisher,
    drain_scheduler_kv_events,
    scheduler_kv_events_to_wire_events,
)
from tokenspeed.runtime.pd.mooncake.entities import ManagerArgs
from tokenspeed.runtime.pd.prefill_executor import DisaggPrefillExecutor
from tokenspeed.runtime.sampling.sampling_params import SamplingParams
from tokenspeed.runtime.utils import (
    configure_logger,
    get_colorful_logger,
    get_zmq_socket,
)
from tokenspeed.runtime.utils.exceptions import get_exception_traceback
from tokenspeed.runtime.utils.nvtx import nvtx_range
from tokenspeed.runtime.utils.process import register_usr_signal
from tokenspeed.runtime.utils.server_args import PortArgs, ServerArgs
from tokenspeed.runtime.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = get_colorful_logger(__name__)


def calc_l3_query_hashes(scheduler, tokens: list[int]) -> list[str]:
    return scheduler.calc_rolling_hash(tokens, apply_match=True)


# Sleep between iterations while frozen (PAUSED_ALL) so the keep-mode pause does
# not busy-spin a CPU core waiting for /resume.
_PAUSED_IDLE_SLEEP_S = 0.001


def _forward_op_executes_model_forward(forward_op, *, is_disagg_decode: bool) -> bool:
    """Return whether ``forward_op`` will enter the model forward path.

    On decode-side PD, EXTEND ops only start remote KV receive; the model
    forward runs after the remote prefill completes and the scheduler advances
    the request into decode. Treating those EXTEND ops as model work makes
    idle DP ranks enter dummy collectives that the active rank will not match.
    """
    if forward_op is None:
        return False
    if sum(forward_op.input_lengths) <= 0:
        return False
    if is_disagg_decode and forward_op.num_extends() > 0:
        return False
    return True


class _NullSender:
    """No-op ZMQ sender for non-rank-0 workers."""

    @staticmethod
    def send_pyobj(x):
        return None


@dataclass(frozen=True)
class DpForwardMetadata:
    global_num_tokens: list[int]
    global_batch_size: list[int]
    global_forward_mode: list[int]
    all_decode_or_idle: bool
    need_idle_forward: bool


class EventLoop:
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        attn_tp_rank: int,
        dp_rank: int,
        global_rank: int,
    ) -> None:
        # Do not pass server_args further down the stack after this point.

        self.server_args = server_args
        self.port_args = port_args
        self.gpu_id = gpu_id
        self.global_rank = global_rank

        self.model_config = self._load_model_config(server_args.model)
        if server_args.speculative_draft_model_path is not None:
            draft_model_config = self._load_model_config(
                server_args.speculative_draft_model_path,
                is_draft_worker=True,
            )
        else:
            draft_model_config = None

        min_per_gpu_mem = self._init_distributed()

        target, draft = create_model_runner(
            server_args, self.model_config, draft_model_config, gpu_id, global_rank
        )

        # ----------------------------------------------------------------
        # HiMA Phase 2 — pre-construction arena factory.
        #
        # When XPool dynamic capacity is enabled the VMM arenas must be
        # created *before* pool construction so that k/v and mamba tensors
        # are born on stable VMM-backed VA.  CUDA graphs then capture those
        # VA pointers; later XPool fires remap physical handles while the
        # VA (and therefore the captured pointers) remain constant.
        #
        # The factory closure is passed into create_attn_components and
        # called inside _create_hybrid_linear_attn right after profiling
        # determines the pool sizes.  On success it stores the arenas in
        # self._pre_built_xpool_arenas for _init_xpool_actuator to reuse.
        # On any failure it returns (None, None) and pools fall back to the
        # normal torch.zeros path.
        # ----------------------------------------------------------------
        self._pre_built_xpool_arenas: "tuple | None" = None

        _enable_xpool = (
            getattr(server_args, "enable_xpool_dynamic_capacity", False)
            and attn_tp_rank == 0
        )
        # Detect hybrid (mamba) models before pool creation; mirrors the
        # has_mamba logic computed after create_attn_components below.
        _hf_cfg_pre = getattr(self.model_config, "hf_config", None)
        _txt_cfg_pre = (
            getattr(_hf_cfg_pre, "text_config", None) if _hf_cfg_pre else None
        )
        _pre_has_mamba = getattr(
            self.model_config, "mambaish_config", None
        ) is not None or (
            _txt_cfg_pre is not None and hasattr(_txt_cfg_pre, "mamba2_cache_params")
        )

        # HiMA Phase 3 (S2.2-followup): resolve the Mamba pool headroom *up
        # front*, before the pool is constructed. The Python SimpleMambaPool
        # tensor must be sized to (base + headroom + 1 + draft) slots from
        # the start because changing the row-dim post-CUDA-graph-capture
        # would invalidate captured strides.  We feed the same value to the
        # C++ scheduler (mamba_pool_total_chunks = base + headroom) and to
        # the registry (mamba_pool_size = base + headroom + 1).  Auto-derive
        # uses budgeter_pages_per_fire because the byte-balanced formula
        # needs the post-pool mamba_bytes_per_slot, which doesn't exist yet.
        self._xpool_mamba_headroom_slots = 0
        if _enable_xpool and _pre_has_mamba:
            cfg_headroom = int(getattr(server_args, "xpool_mamba_headroom_slots", 0))
            if cfg_headroom > 0:
                self._xpool_mamba_headroom_slots = cfg_headroom
            else:
                _ppf = max(1, int(getattr(server_args, "budgeter_pages_per_fire", 64)))
                # Conservative slot-count default that scales with fire size
                # but stays small enough that the permanently-reserved Mamba
                # memory is < 100 MB for typical hybrid models.  For Qwen3.5
                # (mamba_bytes_per_slot ≈ 2 MB) this caps the reserved
                # headroom at ~64 MB.  Override via ServerArgs to allow more
                # cross-pool oscillation before fires hit the slot ceiling.
                self._xpool_mamba_headroom_slots = max(_ppf // 4, 32)

        _xpool_arena_factory = None
        if _enable_xpool and _pre_has_mamba:
            _xpool_arena_factory = self._make_xpool_pre_arena_factory(server_args)

        (
            attn_backend,
            token_to_kv_pool,
            draft_attn_backend,
            draft_token_to_kv_pool,
            self.max_total_num_tokens,
            mamba_pool_total_chunks,
            mamba_pool,
        ) = create_attn_components(
            server_args,
            self.model_config,
            gpu_id,
            global_rank,
            min_per_gpu_mem,
            server_args.enable_memory_saver,
            draft_model_config,
            _xpool_arena_factory=_xpool_arena_factory,
            xpool_mamba_headroom_slots=self._xpool_mamba_headroom_slots,
        )

        num_total_pages = self.max_total_num_tokens // server_args.block_size
        hf_config = getattr(self.model_config, "hf_config", None)
        text_config = getattr(hf_config, "text_config", None) if hf_config else None
        has_mamba = getattr(self.model_config, "mambaish_config", None) is not None or (
            text_config is not None and hasattr(text_config, "mamba2_cache_params")
        )

        model_executor_config = ModelExecutorConfig.from_server_args(
            server_args=server_args,
            model_config=self.model_config,
            max_req_pool_size=server_args.max_num_seqs,
            gpu_id=gpu_id,
            global_rank=global_rank,
            num_total_pages=num_total_pages,
        )
        self.model_executor = create_model_executor(
            server_args=server_args,
            config=model_executor_config,
            model_runner=target,
            draft_model_runner=draft,
            attn_backend=attn_backend,
            token_to_kv_pool=token_to_kv_pool,
            draft_attn_backend=draft_attn_backend,
            draft_token_to_kv_pool=draft_token_to_kv_pool,
            mamba_pool=mamba_pool,
        )

        # Reserve one token slot because request validation uses a strict
        # ``< max_req_len`` check against the model context length.
        self.max_req_input_len = self.model_config.context_len - 1
        mapping = server_args.mapping
        self.attn_tp_size = server_args.attn_tp_size or mapping.attn.tp_size
        self.world_size = server_args.world_size or mapping.world_size
        self.attn_tp_rank = attn_tp_rank
        self.attn_tp_cpu_group = pg_manager.get_process_group(
            "gloo", server_args.mapping.attn.tp_group
        )
        self._pending_cache_event_payloads: OrderedDict[tuple[str, int], dict] = (
            OrderedDict()
        )
        # All ranks submit identical cache plans (the C++ scheduler is mirrored),
        # so a local in-flight counter mirrors across ranks: if it's 0 here, no
        # rank has anything pending. Lets us skip the TP collective in
        # _commit_cache_results entirely when nothing is in flight.
        self._num_inflight_cache_ops = 0
        self.dp_rank = dp_rank
        self.dp_size = mapping.attn.dp_size
        self.has_dp = mapping.has_attn_dp
        if self.has_dp:
            self.world_cpu_group = pg_manager.get_process_group(
                "gloo", mapping.world_group
            )
            self._dp_local_info = torch.zeros(1, 3, dtype=torch.int32)
            self._dp_global_info = torch.zeros(mapping.world_size, 3, dtype=torch.int32)
        if not server_args.enable_kvstore:
            logger.warning(
                "KVStore L2 cache will not be used during normal execution, but it will still be used when retraction happens."
            )

        mamba_l2_host_slots = 0
        if has_mamba and server_args.enable_mamba_l2:
            if server_args.mamba_l2_host_slots > 0:
                mamba_l2_host_slots = server_args.mamba_l2_host_slots
            elif server_args.mamba_l2_host_gb > 0 and mamba_pool is not None:
                slot_bytes = int(
                    mamba_pool.conv_state.shape[0]
                    * (
                        mamba_pool.conv_state[0, 0].nbytes
                        + mamba_pool.ssm_state[0, 0].nbytes
                    )
                )
                mamba_l2_host_slots = int(
                    server_args.mamba_l2_host_gb * (1024**3) // max(slot_bytes, 1)
                )
            else:
                mamba_l2_host_slots = max(
                    int(mamba_pool_total_chunks * server_args.mamba_l2_ratio), 1
                )

        mem_cfg = MemoryExecutorConfig(
            layer_num=self.model_config.num_hidden_layers,
            page_size=server_args.block_size,
            host_ratio=server_args.kvstore_ratio,
            host_size_gb=server_args.kvstore_size,
            io_backend=server_args.kvstore_io_backend,
            host_layout=server_args.kvstore_mem_layout,
            storage_backend=server_args.kvstore_storage_backend,
            storage_backend_extra_config=server_args.kvstore_storage_backend_extra_config,
            model_name=server_args.model,
            enable_mamba_l2=server_args.enable_mamba_l2,
            mamba_l2_host_slots=mamba_l2_host_slots,
            mamba_l2_layout=server_args.mamba_l2_layout,
            mamba_l2_io_backend=server_args.mamba_l2_io_backend,
        )
        if not token_to_kv_pool.supports_hierarchical_kv_cache:
            if server_args.enable_kvstore:
                raise NotImplementedError(
                    "This KV cache pool does not support hierarchical cache "
                    "(kvstore); pass --disable-kvstore."
                )
            self.memory_executor = None
            num_host_pages = 0
        else:
            self.memory_executor = MemoryExecutor(
                device_pool=token_to_kv_pool,
                config=mem_cfg,
                is_dp_attention_enabled=self.has_dp,
                tp_group=self.attn_tp_cpu_group,
                draft_device_pool=draft_token_to_kv_pool,
                mamba_pool=mamba_pool,
            )
            num_host_pages = self.memory_executor.host_pool.page_num

        # For DP attention, max_batch_size must be per-rank to avoid
        # req_pool_allocator overflow.  The C++ scheduler allocates
        # req_pool_slots based on this value, so it must match the
        # per-DP-rank budget (same division used in cuda_graph_wrapper).
        per_rank_max_batch = server_args.max_num_seqs // max(self.dp_size, 1)
        self._kv_events_enabled = (
            EventPublisherFactory.is_enabled(server_args.kv_events_config)
            and attn_tp_rank == 0
        )

        if has_mamba and server_args.max_mamba_cache_size is None:
            logger.info(
                f"Mamba radix cache enabled without explicit max_mamba_cache_size. "
                f"Auto-derived mamba_pool_total_chunks={mamba_pool_total_chunks} "
                f"(ratio={server_args.mamba_full_memory_ratio})."
            )

        # Adjunct enabled only when pool opts in AND prefix-caching switch is on.
        paged_cache_groups = pool_to_paged_cache_groups(token_to_kv_pool)
        self._paged_cache_groups = paged_cache_groups
        prefix_cache_adjunct = None
        required_groups = token_to_kv_pool.prefix_cache_required_group_ids
        if required_groups is not None and server_args.enable_prefix_caching:
            prefix_cache_adjunct = pool_to_prefix_cache_adjunct_spec(required_groups)
        kv_bytes_per_page, mamba_bytes_per_slot = compute_lpb_byte_sizes(
            token_to_kv_pool,
            server_args.block_size,
            mamba_pool=mamba_pool,
        )
        # Profiled KV page count — this is the initial (baseline) KV capacity.
        profiled_kv_pages = self.max_total_num_tokens // server_args.block_size

        # When XPool dynamic capacity is active the VA window needs headroom
        # above the profiled baseline so transfers in both directions have room.
        # Reserve 4× budgeter_pages_per_fire extra pages in each direction.
        # Mamba headroom (in slots) was resolved up front so the Python pool
        # was sized at base + headroom; reuse it here so the C++ allocator's
        # mamba slot bound matches the Python tensor's row dimension.
        xpool_pages_headroom = 0
        xpool_mamba_headroom = self._xpool_mamba_headroom_slots
        if getattr(server_args, "enable_xpool_dynamic_capacity", False):
            pages_per_fire = int(getattr(server_args, "budgeter_pages_per_fire", 64))
            xpool_pages_headroom = pages_per_fire * 4

        scheduler_cfg = make_config(
            num_device_pages=profiled_kv_pages + xpool_pages_headroom,
            max_scheduled_tokens=server_args.chunked_prefill_size,
            max_batch_size=per_rank_max_batch,
            page_size=server_args.block_size,
            num_host_pages=num_host_pages,
            disable_l2_cache=not server_args.enable_kvstore,
            enable_l3_storage=server_args.kvstore_storage_backend is not None,
            prefetch_threshold=4,  # Keep this hard-coded until it becomes configurable.
            role=server_args.disaggregation_mode,
            enable_kv_cache_events=self._kv_events_enabled,
            decode_input_tokens=(
                server_args.speculative_num_draft_tokens
                if server_args.speculative_algorithm is not None
                else 1
            ),
            disable_prefix_cache=not server_args.enable_prefix_caching,
            enable_mamba=has_mamba,
            mamba_cache_chunk_size=server_args.mamba_cache_chunk_size,
            mamba_pool_total_chunks=mamba_pool_total_chunks + xpool_mamba_headroom,
            enable_mamba_l2=server_args.enable_mamba_l2,
            mamba_l2_host_slots=mamba_l2_host_slots,
            paged_cache_groups=paged_cache_groups,
            enable_mixed_prefill_decode=server_args.enable_mixed_batch,
            prefix_cache_adjunct=prefix_cache_adjunct,
            eviction_policy=server_args.radix_eviction_policy,
            lpb_window_s=server_args.lpb_window_s,
            lpb_hit_deque_maxlen=server_args.lpb_hit_deque_maxlen,
            c_kv_alpha=server_args.csigma_kv_alpha,
            c_kv_beta=server_args.csigma_kv_beta,
            c_kv_gamma=server_args.csigma_kv_gamma,
            c_m=server_args.csigma_m,
            kv_bytes_per_page=kv_bytes_per_page,
            mamba_bytes_per_slot=mamba_bytes_per_slot,
            enable_budgeter=server_args.enable_budgeter,
            enable_admitter=server_args.enable_admitter,
            enable_xpool_dynamic_capacity=server_args.enable_xpool_dynamic_capacity,
            budgeter_tick_s=server_args.budgeter_tick_s,
            budgeter_pages_per_fire=server_args.budgeter_pages_per_fire,
            xpool_nb_margin=server_args.xpool_nb_margin,
            xpool_ewma_tau_s=server_args.xpool_ewma_tau_s,
            xpool_saturation_low=server_args.xpool_saturation_low,
            xpool_reverse_cooldown_s=server_args.xpool_reverse_cooldown_s,
            # PressureAdapter (S2.3): forward-looking signal weights.
            xpool_w_queue=server_args.xpool_w_queue,
            xpool_w_retract=server_args.xpool_w_retract,
            xpool_w_paused=server_args.xpool_w_paused,
            xpool_queue_ref=server_args.xpool_queue_ref,
            xpool_retract_ref=server_args.xpool_retract_ref,
            xpool_paused_ref=server_args.xpool_paused_ref,
            enable_dynamic_admission_cap=server_args.enable_dynamic_admission_cap,
            # Initial capacities: profiled baselines (strictly less than VA window).
            xpool_initial_kv_pages=profiled_kv_pages if xpool_pages_headroom > 0 else 0,
            xpool_initial_mamba_slots=(
                mamba_pool_total_chunks if xpool_mamba_headroom > 0 else 0
            ),
        )
        logger.info(
            "Scheduler config: page_size=%s num_device_pages=%s "
            "max_scheduled_tokens=%s decode_input_tokens=%s disable_l2_cache=%s "
            "max_batch_size=%s (global max_num_seqs=%s, dp_size=%s) "
            "mamba_pool_total_chunks=%s enable_mamba=%s "
            "disable_prefix_cache=%s paged_cache_groups=%s",
            scheduler_cfg.page_size,
            scheduler_cfg.num_device_pages,
            scheduler_cfg.max_scheduled_tokens,
            scheduler_cfg.decode_input_tokens,
            scheduler_cfg.disable_l2_cache,
            scheduler_cfg.max_batch_size,
            server_args.max_num_seqs,
            self.dp_size,
            mamba_pool_total_chunks,
            has_mamba,
            scheduler_cfg.disable_prefix_cache,
            [group.group_id for group in paged_cache_groups],
        )
        self.scheduler = Scheduler(scheduler_cfg)
        self._budgeter_tick_s = float(server_args.budgeter_tick_s)
        self._last_budget_tick = 0.0
        # Replay metrics: when TOKENSPEED_REPLAY_METRICS_PATH is set to a file
        # path, the event loop appends a JSONL snapshot every
        # TOKENSPEED_REPLAY_METRICS_INTERVAL_S seconds (default 0.1). This is
        # the per-tick time series that ``tokenspeed.agentreplay`` uses to
        # plot pool occupancy / fire timing for HiMA A/B benchmarks. Keeping
        # the contract here (rather than in agentreplay) means any harness
        # can opt in by setting the env var, not just the replay CLI.
        self._replay_metrics_path = os.environ.get("TOKENSPEED_REPLAY_METRICS_PATH")
        self._replay_metrics_interval_s = float(
            os.environ.get("TOKENSPEED_REPLAY_METRICS_INTERVAL_S", "0.1")
        )
        self._replay_metrics_file = None
        self._replay_metrics_last_t = 0.0
        self._replay_metrics_start_t = 0.0
        if self._replay_metrics_path:
            # Open in line-buffered mode so a crash doesn't lose the recent
            # tail. We intentionally truncate any prior file so each run
            # has its own time series.
            self._replay_metrics_file = open(
                self._replay_metrics_path, "w", buffering=1
            )
        self._init_xpool_actuator(
            server_args,
            has_mamba,
            token_to_kv_pool=token_to_kv_pool,
            kv_bytes_per_page=kv_bytes_per_page,
            profiled_kv_pages=profiled_kv_pages,
        )
        token_to_kv_pool.bind_paged_cache_scheduler(self.scheduler)
        if attn_tp_rank == 0:
            self.kv_event_publisher = EventPublisherFactory.create(
                server_args.kv_events_config,
                attn_dp_rank=dp_rank,
            )
        else:
            self.kv_event_publisher = NullEventPublisher(attn_dp_rank=dp_rank)

        self._init_interprocess_comm()

        # Pause/resume control state. Shared with the request handler, which
        # drives the control-request side; the event loop reads the gate.
        self._pause = PauseController(self.send_to_tokenizer)

        # GPU-memory data plane (release/resume_memory_occupation). Reuses the
        # pause controller's drain machinery; frees memory via the memory-saver
        # adapter once the scheduler drains. See memory_occupation.py.
        # Releasing KV is only safe if any prefix cache it backs can be cleared:
        # either prefix caching is off, or the scheduler exposes a reset. Decide
        # once here (static config) and let the controller reject unsafe releases.
        kv_cache_release_allowed = (
            not self.server_args.enable_prefix_caching
            or callable(getattr(self.scheduler, "reset_prefix_cache", None))
        )
        self._memory = MemoryOccupationController(
            send_func=self.send_to_tokenizer,
            pause_controller=self._pause,
            adapter=TorchMemorySaverAdapter.create(
                enable=self.server_args.enable_memory_saver
            ),
            enabled=self.server_args.enable_memory_saver,
            reset_caches_fn=self._reset_caches_for_release,
            kv_repair_fn=self._kv_repair_after_wake,
            kv_cache_release_allowed=kv_cache_release_allowed,
        )

        self.metrics = EngineMetrics(
            labels={
                "model_name": server_args.served_model_name,
                "app_key": server_args.app_key or "",
                "dp_rank": str(dp_rank),
            },
            enabled=(
                server_args.enable_metrics
                and attn_tp_rank == 0
                and "prometheus" in (server_args.metrics_reporters or [])
            ),
        )

        self.request_handler = RequestHandler(
            server_args=self.server_args,
            hf_eos_token_id=self.model_config.hf_eos_token_id,
            max_req_len=self.model_config.context_len - 1,
            vocab_size=self.model_config.vocab_size,
            recv_func=self.recv_from_tokenizer,
            send_func=self.send_to_tokenizer,
            get_load_fn=self._get_load,
            architectures=self.model_config.hf_config.architectures,
            pause_controller=self._pause,
            memory_controller=self._memory,
        )

        self.output_processor = OutputProcesser(
            send_to_tokenizer=self.send_to_tokenizer,
            global_rank=global_rank,
            spec_algorithm=self.server_args.speculative_algorithm,
            spec_num_tokens=(
                self.server_args.speculative_num_draft_tokens
                if self.server_args.speculative_algorithm is not None
                else None
            ),
            stream_interval=self.server_args.stream_interval,
            metrics=self.metrics,
        )
        self.prefetch_threshold = scheduler_cfg.prefetch_threshold

        if server_args.disaggregation_mode != "null":
            kv_args = get_kv_args(
                global_rank,
                global_rank,
                server_args.disaggregation_ib_device,
                token_to_kv_pool,
                draft_token_to_kv_pool,
                mamba_pool,
            )
            pd_manager_args = ManagerArgs(
                bootstrap_port=server_args.disaggregation_bootstrap_port,
                dist_init_addr=server_args.dist_init_addr,
                world_size=server_args.world_size or mapping.world_size,
                dp_size=server_args.data_parallel_size or mapping.attn.dp_size,
                attn_tp_rank=attn_tp_rank,
                attn_dp_rank=dp_rank,
                is_mla_backend=False,
                draft_is_mla_backend=False,
                enable_metrics=False,
                enable_mla_l1_5_cache=server_args.enable_mla_l1_5_cache,
                served_model_name=server_args.served_model_name,
                app_key=server_args.app_key,
                metrics_reporters=server_args.metrics_reporters,
                enable_dp_attention=self.has_dp,
            )
            self.pd_kv_transfer = create_pd_kv_transfer(
                mode=server_args.disaggregation_mode,
                backend=server_args.disaggregation_transfer_backend,
                args=pd_manager_args,
                kv_args=kv_args,
                gloo_group=self.attn_tp_cpu_group,
                page_size=token_to_kv_pool.page_size,
            )
            self._setup_pd_layerwise_transfer(
                server_args.disaggregation_layerwise_interval
            )
        else:
            self.pd_kv_transfer = None

    def _setup_pd_layerwise_transfer(self, interval: int) -> None:
        if not isinstance(self.pd_kv_transfer, DisaggPrefillExecutor):
            return
        if interval <= 0:
            return

        from tokenspeed.runtime.pd.utils import StepCounter

        step_counter = StepCounter(self.model_executor.device, self.gpu_id)
        self.model_executor.attn_backend.register_step_counter(step_counter)
        if self.model_executor.draft_attn_backend is not None:
            self.model_executor.draft_attn_backend.register_step_counter(step_counter)
        self.pd_kv_transfer.register_layerwise_step_counter(step_counter, interval)

    def _commit_cache_results(self) -> None:
        if self.memory_executor is None:
            return
        cache_results = self.memory_executor.poll_results()
        self._num_inflight_cache_ops -= len(cache_results)
        for event in cache_results:
            payload = cache_event_to_payload(event)
            self._pending_cache_event_payloads[cache_event_key(payload)] = payload

        # Local short-circuit: counter mirrors across ranks (same plan every-
        # where), so if nothing is in flight here AND nothing is buffered
        # awaiting commit, no rank has anything to gather.
        if self._num_inflight_cache_ops == 0 and not self._pending_cache_event_payloads:
            return

        ready_payloads = self._pop_ready_cache_event_payloads()
        if not ready_payloads:
            return
        logger.debug(
            "[cache_poll] got %s synchronized results, advancing scheduler",
            len(ready_payloads),
        )
        ec = ExecutionEvent()
        for payload in ready_payloads:
            e = cache_event_from_payload(payload)
            logger.debug(
                "[cache_poll] event: op_id=%s success=%s type=%s request_id=%s",
                e.op_id,
                e.success,
                type(e).__name__,
                getattr(e, "request_id", "N/A"),
            )
            ec.add_event(e)
        self.scheduler.advance(ec)
        logger.debug("[cache_poll] scheduler.advance() done")
        self._publish_scheduler_kv_events()

    def _publish_scheduler_kv_events(self) -> None:
        raw_events = drain_scheduler_kv_events(
            self.scheduler,
            enabled=self._kv_events_enabled,
        )
        if not raw_events:
            return

        events = scheduler_kv_events_to_wire_events(raw_events)
        if not events:
            return

        self.kv_event_publisher.publish(
            KVEventBatch(ts=time.time(), events=events, attn_dp_rank=self.dp_rank)
        )

    def _pop_ready_cache_event_payloads(self) -> list[dict]:
        local_payloads = list(self._pending_cache_event_payloads.values())
        if self.attn_tp_size == 1:
            ready_payloads = local_payloads
        else:
            gathered_payloads = [None] * self.attn_tp_size
            dist.all_gather_object(
                gathered_payloads,
                local_payloads,
                group=self.attn_tp_cpu_group,
            )
            ready_payloads = pop_common_cache_event_payloads(gathered_payloads)
            if self.attn_tp_rank == 0 and cache_sync_debug_enabled():
                pending_ops = [
                    [(payload["kind"], payload["op_id"]) for payload in rank_payloads]
                    for rank_payloads in gathered_payloads
                ]
                if len({tuple(rank_ops) for rank_ops in pending_ops}) > 1:
                    logger.info(
                        "[cache_sync] rank=%s pending_ops=%s ready_ops=%s",
                        self.global_rank,
                        pending_ops,
                        [
                            (payload["kind"], payload["op_id"])
                            for payload in ready_payloads
                        ],
                    )

        for payload in ready_payloads:
            self._pending_cache_event_payloads.pop(cache_event_key(payload), None)
        return ready_payloads

    def _dispatch_forward(
        self,
        forward_op,
        sampling_params_list,
        execution_plan,
        dp_metadata=None,
        stats=None,
        grammar_inputs=None,
    ):
        """Execute one forward step; return (results, on_first_token).

        results is None when the step produces no model output (Path 2/3).
        Both event_loop and event_loop_overlap call this method; they differ
        only in *when* they call post_process on the returned results.

        Path 1 — no PD:              run forward, return (results, None)
        Path 2 — decode, extend:     trigger RDMA receive, return (None, None)
        Path 3 — prefill, decode:    send KV to decode side, return (None, None)
        Path 4 — prefill, extend:    run prefill forward, return (results, on_first_token)
        """
        if stats is None:
            stats = {}
        dp_global_num_tokens = (
            dp_metadata.global_num_tokens if dp_metadata is not None else None
        )
        dp_global_bs = (
            dp_metadata.global_batch_size if dp_metadata is not None else None
        )
        dp_all_decode_or_idle = (
            dp_metadata.all_decode_or_idle if dp_metadata is not None else False
        )
        multimodal_context = self._get_multimodal_context_for_forward(forward_op)

        self.model_executor.update_block_table(forward_op)

        if self.pd_kv_transfer is None:
            # Path 1: normal (no disaggregation)
            self.model_executor.reset_valid_cache_length(forward_op)
            return (
                self.model_executor.execute_forward_op_with_log(
                    forward_op,
                    sampling_params_list,
                    dp_global_num_tokens=dp_global_num_tokens,
                    dp_global_bs=dp_global_bs,
                    dp_all_decode_or_idle=dp_all_decode_or_idle,
                    grammar_inputs=grammar_inputs,
                    multimodal_context=multimodal_context,
                    **stats,
                ),
                None,
            )

        elif isinstance(self.pd_kv_transfer, DisaggDecodeExecutor):
            # Decode node
            if forward_op.num_extends() > 0:
                # Path 2: new requests waiting for remote KV — trigger RDMA receive
                self.pd_kv_transfer.reset_valid_cache_length(
                    forward_op,
                    self.model_executor.runtime_states,
                    self.model_executor.execution_stream,
                    self.model_executor.device,
                )
                self.pd_kv_transfer.execute(forward_op)
                self.model_executor.reset_remote_prefill_mamba_inputs(forward_op)
                return None, None
            else:
                # Path 3b: decode batch — normal forward
                self.model_executor.reset_valid_cache_length(forward_op)
                return (
                    self.model_executor.execute_forward_op_with_log(
                        forward_op,
                        sampling_params_list,
                        dp_global_num_tokens=dp_global_num_tokens,
                        dp_global_bs=dp_global_bs,
                        dp_all_decode_or_idle=dp_all_decode_or_idle,
                        multimodal_context=multimodal_context,
                        **stats,
                    ),
                    None,
                )

        else:
            # Prefill node (only reached from event_loop, never event_loop_overlap)
            assert isinstance(self.pd_kv_transfer, DisaggPrefillExecutor)
            if forward_op.num_extends() == 0:
                # Path 3: all prefill done — send KV to decode side
                self.pd_kv_transfer.execute(forward_op)
                return None, None
            else:
                # Path 4: extend batch — run prefill forward
                self.model_executor.reset_valid_cache_length(forward_op)
                self.pd_kv_transfer.prepare_prefill(forward_op)
                return (
                    self.model_executor.execute_forward_op_with_log(
                        forward_op,
                        sampling_params_list,
                        dp_global_num_tokens=dp_global_num_tokens,
                        dp_global_bs=dp_global_bs,
                        dp_all_decode_or_idle=dp_all_decode_or_idle,
                        grammar_inputs=grammar_inputs,
                        multimodal_context=multimodal_context,
                        capture_next_input_ids=True,
                        **stats,
                    ),
                    self.pd_kv_transfer.store_prefill_token,
                )

    def _get_multimodal_context_for_forward(self, forward_op):
        if not self.model_config.is_multimodal_active:
            return None

        num_extends = forward_op.num_extends()
        mm_inputs = []
        has_mm = False
        for index, rid in enumerate(forward_op.request_ids):
            state = self.output_processor.rid_to_state.get(rid)
            if state is not None and index < num_extends:
                state.maybe_extend_multimodal_mrope_positions()
            item = getattr(state, "multimodal_inputs", None) if state else None
            mm_inputs.append(item)
            has_mm = has_mm or item is not None
        if not has_mm:
            return None

        from tokenspeed.runtime.multimodal.inputs import MultimodalForwardContext

        return MultimodalForwardContext(
            mm_inputs=mm_inputs,
            extend_prefix_lens=list(forward_op.extend_prefix_lens),
            extend_seq_lens=list(forward_op.input_lengths[:num_extends]),
        )

    def _build_mamba_layerwise_cow(
        self, execution_plan, forward_op
    ) -> dict[int, list[int]]:
        if forward_op is None:
            return {}
        loaded_mamba_slots: set[int] = set()
        for cache_op in execution_plan.cache:
            if not isinstance(cache_op, Cache.LoadBackOp):
                continue
            dst_by_kind = getattr(cache_op, "dst_pages_by_kind", None)
            if dst_by_kind is None:
                dst_groups = getattr(cache_op, "dst_pages", [])
            else:
                dst_groups = dst_by_kind.get(CacheKind.MAMBA.value, [])
            for dst_pages in dst_groups:
                loaded_mamba_slots.update(int(page) for page in dst_pages)
        if not loaded_mamba_slots:
            return {}

        cow_src_indices = getattr(forward_op, "mamba_cow_src_indices", None)
        working_indices = getattr(forward_op, "mamba_pool_indices", None)
        if cow_src_indices is None or working_indices is None:
            return {}

        cow_by_src: dict[int, list[int]] = {}
        for cow_src, working in zip(list(cow_src_indices), list(working_indices)):
            cow_src = int(cow_src)
            working = int(working)
            if cow_src < 0 or working < 0 or cow_src not in loaded_mamba_slots:
                continue
            cow_dsts = cow_by_src.setdefault(cow_src, [])
            if working not in cow_dsts:
                cow_dsts.append(working)
        return cow_by_src

    def _submit_cache_ops(self, execution_plan) -> None:
        if self.memory_executor is None:
            return
        forward_op = self._get_forward_op(execution_plan)
        mamba_layerwise_cow = self._build_mamba_layerwise_cow(
            execution_plan, forward_op
        )
        if mamba_layerwise_cow:
            self.model_executor.set_layerwise_mamba_cow_done(mamba_layerwise_cow)
            self.memory_executor.set_mamba_layerwise_cow(mamba_layerwise_cow)
        self.memory_executor.submit_plan(execution_plan)
        for op in execution_plan.cache:
            if isinstance(op, Cache.WriteBackOp):
                self._num_inflight_cache_ops += len(op.op_ids)
            elif isinstance(op, Cache.LoadBackOp):
                continue
            elif isinstance(op, (Cache.PrefetchOp, Cache.BackUpOp)):
                self._num_inflight_cache_ops += 1
            else:
                raise ValueError(f"unsupported cache op kind: {type(op).__name__}")
        self._setup_layerwise_loadback(execution_plan)

    def _setup_layerwise_loadback(self, execution_plan) -> None:
        host_exec = getattr(self.memory_executor, "host_exec", None)
        available_pools = (
            getattr(host_exec, "pools", {}) if host_exec is not None else {}
        )
        consumer_indices_by_kind: dict[CacheKind, list[int]] = {
            kind: [] for kind in available_pools
        }
        for cache_op in execution_plan.cache:
            if isinstance(cache_op, Cache.LoadBackOp):
                for op_id in cache_op.op_ids:
                    for kind in consumer_indices_by_kind:
                        producer_idx = self.memory_executor.get_producer_index(
                            kind, op_id
                        )
                        if (
                            producer_idx is not None
                            and producer_idx not in consumer_indices_by_kind[kind]
                        ):
                            consumer_indices_by_kind[kind].append(producer_idx)
        for kind, consumer_indices in consumer_indices_by_kind.items():
            self.memory_executor.set_consumer(
                kind, consumer_indices if consumer_indices else -1
            )

    def _flush_mamba_retract_states(self, forward_op) -> None:
        """Copy draft->working mamba states when retract occurred (no forward scheduled)."""
        if forward_op is not None:
            return
        if self.model_executor.drafter is None:
            return
        if self.model_executor.runtime_states.mamba_pool is None:
            return
        self.model_executor.flush_mamba_draft_to_working_on_retract()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_model_config(
        self, model_path: str, is_draft_worker: bool = False
    ) -> ModelConfig:
        server_args = self.server_args
        quantization = server_args.quantization
        if is_draft_worker:
            quantization = server_args.speculative_draft_model_quantization
        return ModelConfig(
            model_path,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
            context_length=server_args.max_model_len,
            model_override_args=server_args.hf_overrides,
            dtype=server_args.dtype,
            quantization=quantization,
            server_args=server_args,
            is_draft_worker=is_draft_worker,
        )

    def _init_distributed(self) -> float:
        max_num_input_tokens = (
            self.server_args.chunked_prefill_size
            if self.server_args.chunked_prefill_size > 0
            else self.server_args.max_prefill_tokens + self.server_args.max_model_len
        )
        distributed_config = DistributedConfig.from_server_args(
            server_args=self.server_args,
            port_args=self.port_args,
            gpu_id=self.gpu_id,
            global_rank=self.global_rank,
            hidden_size=self.model_config.hidden_size,
            max_num_tokens=max_num_input_tokens,
        )
        return DistributedInitializer.initialize(distributed_config)

    def _init_interprocess_comm(self):
        context = zmq.Context(2)
        if self.attn_tp_rank == 0:
            self.recv_from_tokenizer = get_zmq_socket(
                context, zmq.PULL, self.port_args.scheduler_input_ipc_name, False
            )
            self.send_to_tokenizer = get_zmq_socket(
                context, zmq.PUSH, self.port_args.tokenizer_ipc_name, False
            )
        else:
            self.recv_from_tokenizer = None
            self.send_to_tokenizer = _NullSender()

    # ------------------------------------------------------------------
    # Shared step helpers
    # ------------------------------------------------------------------

    def _reap_or_keep_buffered_spec(self, spec) -> bool:
        """Resolve a buffered spec on resume; return True if it should be admitted.

        A buffered spec was already registered in ``rid_to_state`` before it was
        withheld, so if it was aborted while paused it never reached the
        scheduler and the forward path can never reap it. Handle that here:

        - state missing  -> already published and reaped; drop silently.
        - state finished -> aborted in place. Stream a terminating finish for
          pause-initiated aborts (the passive client is still waiting) and drop
          the registered state so the rid does not leak; client-initiated aborts
          already tore down their own state, so just reap.
        - otherwise      -> still live; admit it.
        """
        state = self.output_processor.rid_to_state.get(spec.request_id)
        if state is None:
            return False
        if state.finished:
            if state.abort_notify_client:
                self.output_processor.publish_finished_at_admission(
                    spec.request_id, state
                )
            else:
                self.output_processor.rid_to_state.pop(spec.request_id, None)
            return False
        return True

    def _make_xpool_pre_arena_factory(self, server_args) -> "callable":
        """Return a factory closure for pre-construction VMM arenas (HiMA Phase 2).

        The returned callable is passed to ``create_attn_components`` and
        invoked inside ``_create_hybrid_linear_attn`` *after* profiling
        determines pool sizes but *before* any tensor is allocated.  On
        success it stores the arenas in ``self._pre_built_xpool_arenas`` and
        returns them; on failure it raises so the caller falls back to the
        normal ``torch.zeros`` path.
        """
        import math

        import torch

        from tokenspeed.runtime.cache.arena._cuda_vmm import CHUNK_SIZE_BYTES
        from tokenspeed.runtime.cache.arena.chunk_arena import ChunkArena
        from tokenspeed.runtime.cache.arena.kv_arena import KvLayerArenaGroup
        from tokenspeed.runtime.cache.arena.shared_pool import SharedHandlePool

        gpu_id = self.gpu_id
        pages_per_fire = max(
            1, int(getattr(server_args, "budgeter_pages_per_fire", 64))
        )

        def _factory(
            *,
            kv_num_layers: int,
            kv_max_tokens: int,
            kv_page_size: int,
            kv_head_num: int,
            kv_head_dim: int,
            kv_dtype_itemsize: int,
            mamba_total_size: int,
            mamba_num_layers: int,
            mamba_conv_shape: tuple,
            mamba_ssm_shape: tuple,
            mamba_conv_dtype: "torch.dtype",
            mamba_ssm_dtype: "torch.dtype",
        ) -> tuple:
            # ----------------------------------------------------------
            # Mamba arena: sized to hold conv_state + ssm_state for the
            # full mamba pool plus headroom for incoming KV→Mamba fires.
            # ----------------------------------------------------------
            conv_bytes = (
                math.prod((mamba_num_layers, mamba_total_size, *mamba_conv_shape))
                * torch.empty(0, dtype=mamba_conv_dtype).element_size()
            )
            ssm_bytes = (
                math.prod((mamba_num_layers, mamba_total_size, *mamba_ssm_shape))
                * torch.empty(0, dtype=mamba_ssm_dtype).element_size()
            )
            mamba_pool_bytes = conv_bytes + ssm_bytes
            mamba_current_chunks = max(
                1, math.ceil(mamba_pool_bytes / CHUNK_SIZE_BYTES)
            )
            # HiMA Phase 3 (S2.2-followup): mamba_total_size already accounts
            # for xpool_mamba_headroom_slots (see _create_hybrid_linear_attn).
            # The arena is sized at base+headroom and FULLY pre-mapped at
            # boot so the SimpleMambaPool tensor (a view over the same VA)
            # is backed by physical pages from slot 0 up to total_size.
            # The XPoolActuator detects the equal max_chunks/mapped_chunks
            # signal and treats Mamba as logical-only — kv_to_mamba and
            # mamba_to_kv fires update the C++ allocator's slot bound but
            # never resize the Mamba arena, which the conv/ssm
            # contiguous-slice layout cannot safely support.
            mamba_max_chunks = mamba_current_chunks
            mamba_mapped_chunks = mamba_current_chunks

            mamba_handles = SharedHandlePool(mamba_max_chunks, device=gpu_id)
            mamba_arena = ChunkArena(
                shared_pool=mamba_handles,
                max_chunks=mamba_max_chunks,
                mapped_chunks=mamba_mapped_chunks,
                name="xpool_mamba",
            )
            logger.info(
                "XPool pre-arenas factory: mamba arena created "
                "(%d/%d chunks pre-mapped, %d bytes; logical-only resize)",
                mamba_mapped_chunks,
                mamba_max_chunks,
                mamba_pool_bytes,
            )

            # ----------------------------------------------------------
            # KV per-layer arena group: one arena per layer k/v buffer.
            # Pre-construction maps the *full* tensor (all rows) so there
            # is no doubling — only the VMM physical handles are used from
            # the start.  The tail of the VA window is still reserved for
            # future mamba_to_kv growth.
            # ----------------------------------------------------------
            total_rows = kv_max_tokens + kv_page_size
            single_layer_bytes = (
                total_rows * kv_head_num * kv_head_dim * kv_dtype_itemsize
            )
            # All rows live: map the full capacity upfront.
            max_chunks_per_layer = max(
                1, math.ceil(single_layer_bytes / CHUNK_SIZE_BYTES)
            )
            # Reserve extra VA for mamba_to_kv headroom (up to 4× fire size).
            kv_headroom_pages = pages_per_fire * 4
            kv_headroom_bytes = (
                kv_headroom_pages
                * kv_page_size
                * kv_head_num
                * kv_head_dim
                * kv_dtype_itemsize
            )
            kv_headroom_chunks = max(1, math.ceil(kv_headroom_bytes / CHUNK_SIZE_BYTES))
            kv_max_chunks_with_headroom = max_chunks_per_layer + kv_headroom_chunks

            k_arenas: list = []
            v_arenas: list = []
            for li in range(kv_num_layers):
                k_pool = SharedHandlePool(max_chunks_per_layer, device=gpu_id)
                v_pool = SharedHandlePool(max_chunks_per_layer, device=gpu_id)
                k_arenas.append(
                    ChunkArena(
                        shared_pool=k_pool,
                        max_chunks=kv_max_chunks_with_headroom,
                        mapped_chunks=max_chunks_per_layer,
                        name=f"xpool_kv_k{li}",
                    )
                )
                v_arenas.append(
                    ChunkArena(
                        shared_pool=v_pool,
                        max_chunks=kv_max_chunks_with_headroom,
                        mapped_chunks=max_chunks_per_layer,
                        name=f"xpool_kv_v{li}",
                    )
                )

            kv_layer_group = KvLayerArenaGroup(
                k_arenas=k_arenas,
                v_arenas=v_arenas,
                page_size=kv_page_size,
                head_num=kv_head_num,
                head_dim=kv_head_dim,
                dtype_itemsize=kv_dtype_itemsize,
                initial_live_rows=total_rows,  # all rows live for pre-construction
            )
            logger.info(
                "XPool pre-arenas factory: KV layer group created "
                "(%d layers, %d chunks/layer, total_rows=%d)",
                kv_num_layers,
                max_chunks_per_layer,
                total_rows,
            )

            # Persist for _init_xpool_actuator to pick up after
            # create_model_executor (CUDA graph capture) completes.
            self._pre_built_xpool_arenas = (kv_layer_group, mamba_arena)
            return kv_layer_group, mamba_arena

        return _factory

    def _init_xpool_actuator(
        self,
        server_args,
        has_mamba: bool,
        *,
        token_to_kv_pool=None,
        kv_bytes_per_page: int = 0,
        profiled_kv_pages: int = 0,
    ) -> None:
        """Set up the inter-pool capacity-transfer actuator (HiMA Phase 2).

        Only created when ``--enable-xpool-dynamic-capacity`` is set, on the
        attention TP leader, for hybrid (mamba) models.

        When ``self._pre_built_xpool_arenas`` is set (pre-construction path),
        the arenas were already created before pool construction and CUDA graph
        capture, so pool tensors are already on stable VMM VA.  This method
        then just wires up the ``XPoolActuator`` without any rebinding.

        Otherwise (legacy post-capture path), arenas are created here and
        ``bind_arena`` is attempted for KV (skipped for Mamba due to CUDA
        graph stale-pointer risk).
        """
        self._xpool_actuator = None
        if not getattr(server_args, "enable_xpool_dynamic_capacity", False):
            return
        if self.attn_tp_rank != 0 or not has_mamba:
            return
        try:
            import math

            import torch.cuda

            from tokenspeed.runtime.cache.arena._cuda_vmm import CHUNK_SIZE_BYTES
            from tokenspeed.runtime.cache.arena.chunk_arena import ChunkArena
            from tokenspeed.runtime.cache.arena.kv_arena import KvLayerArenaGroup
            from tokenspeed.runtime.cache.arena.shared_pool import SharedHandlePool
            from tokenspeed.runtime.cache.arena.xpool_actuator import XPoolActuator

            pages_per_fire = max(
                1, int(getattr(server_args, "budgeter_pages_per_fire", 64))
            )

            # ----------------------------------------------------------------
            # Fast path: pre-construction arenas were built by the factory
            # before pool construction and CUDA graph capture.  Pool tensors
            # are already on stable VMM VA — no rebinding needed.
            # ----------------------------------------------------------------
            if self._pre_built_xpool_arenas is not None:
                kv_arena, mamba_arena = self._pre_built_xpool_arenas
                mamba_current_chunks = getattr(mamba_arena, "mapped_chunks", 0)
                mamba_max_chunks = getattr(mamba_arena, "max_chunks", 0)
                self._xpool_actuator = XPoolActuator(
                    kv_arena=kv_arena,
                    mamba_arena=mamba_arena,
                    scheduler=self.scheduler,
                    kv_bytes_per_page=kv_bytes_per_page,
                )
                logger.info(
                    "XPool dynamic capacity enabled (pre-construction path): "
                    "pages_per_fire=%d mamba_chunks=%d/%d "
                    "kv_bound=True (VMM VA, physical transfer enabled) "
                    "kv_bytes_per_page=%d",
                    pages_per_fire,
                    mamba_current_chunks,
                    mamba_max_chunks,
                    kv_bytes_per_page,
                )
                return

            # ----------------------------------------------------------------
            # Legacy post-capture path: arenas were not pre-built (e.g. the
            # model is not a hybrid GDN, XPool was enabled late, or the factory
            # failed).  Create arenas now and attempt KV rebinding; skip Mamba
            # rebinding to avoid CUDA-graph stale-pointer issues.
            # ----------------------------------------------------------------
            rs = getattr(self.model_executor, "runtime_states", None)
            mamba_pool = getattr(rs, "mamba_pool", None) if rs is not None else None

            mamba_pool_bytes = 0
            if mamba_pool is not None:
                conv = mamba_pool.conv_state
                ssm = mamba_pool.ssm_state
                mamba_pool_bytes = conv.nbytes + ssm.nbytes

            mamba_current_chunks = (
                max(1, math.ceil(mamba_pool_bytes / CHUNK_SIZE_BYTES))
                if mamba_pool_bytes > 0
                else pages_per_fire * 4
            )

            # HiMA Phase 3 (S2.2-followup): legacy post-capture path also
            # treats Mamba as logical-only.  The Python SimpleMambaPool
            # tensor is sized at base+headroom (via
            # xpool_mamba_headroom_slots passed to the registry), so the
            # C++ allocator can grow logically without overflowing the
            # tensor.  The arena's physical mapping never moves —
            # set max_chunks == mapped_chunks to signal the static state
            # to XPoolActuator._mamba_is_static.
            mamba_max_chunks = mamba_current_chunks

            mamba_handles = SharedHandlePool(mamba_max_chunks, device=self.gpu_id)
            mamba_arena = ChunkArena(
                shared_pool=mamba_handles,
                max_chunks=mamba_max_chunks,
                mapped_chunks=mamba_current_chunks,
                name="xpool_mamba",
            )

            # ----------------------------------------------------------------
            # KV per-layer arena group (legacy path — post-capture rebind).
            # ----------------------------------------------------------------
            kv_arena: ChunkArena | KvLayerArenaGroup | None = None
            kv_bound = False

            from tokenspeed.runtime.layers.attention.kv_cache.mha import (
                MHATokenToKVPool,
            )

            _kv_inner = getattr(token_to_kv_pool, "inner", token_to_kv_pool)

            if (
                token_to_kv_pool is not None
                and isinstance(_kv_inner, MHATokenToKVPool)
                and profiled_kv_pages > 0
            ):
                try:
                    pool = _kv_inner
                    page_size = pool.page_size
                    head_num = pool.head_num
                    head_dim = pool.head_dim
                    layer_num = pool.layer_num
                    dtype_itemsize = torch._utils._element_size(pool.store_dtype)

                    total_rows = pool.size + page_size
                    initial_live_rows = min(
                        (profiled_kv_pages + 1) * page_size, total_rows
                    )

                    single_layer_bytes = (
                        total_rows * head_num * head_dim * dtype_itemsize
                    )
                    max_chunks_per_layer = max(
                        1, math.ceil(single_layer_bytes / CHUNK_SIZE_BYTES)
                    )
                    live_bytes_per_layer = (
                        initial_live_rows * head_num * head_dim * dtype_itemsize
                    )
                    initial_chunks_per_layer = max(
                        1, math.ceil(live_bytes_per_layer / CHUNK_SIZE_BYTES)
                    )

                    k_arenas: list[ChunkArena] = []
                    v_arenas: list[ChunkArena] = []
                    for li in range(layer_num):
                        k_pool = SharedHandlePool(
                            initial_chunks_per_layer, device=self.gpu_id
                        )
                        v_pool = SharedHandlePool(
                            initial_chunks_per_layer, device=self.gpu_id
                        )
                        k_arenas.append(
                            ChunkArena(
                                shared_pool=k_pool,
                                max_chunks=max_chunks_per_layer,
                                mapped_chunks=initial_chunks_per_layer,
                                name=f"xpool_kv_k{li}",
                            )
                        )
                        v_arenas.append(
                            ChunkArena(
                                shared_pool=v_pool,
                                max_chunks=max_chunks_per_layer,
                                mapped_chunks=initial_chunks_per_layer,
                                name=f"xpool_kv_v{li}",
                            )
                        )

                    kv_layer_group = KvLayerArenaGroup(
                        k_arenas=k_arenas,
                        v_arenas=v_arenas,
                        page_size=page_size,
                        head_num=head_num,
                        head_dim=head_dim,
                        dtype_itemsize=dtype_itemsize,
                        initial_live_rows=initial_live_rows,
                    )

                    torch.cuda.synchronize()
                    pool.bind_arena(kv_layer_group)
                    kv_arena = kv_layer_group
                    kv_bound = True
                    logger.info(
                        "XPool: KV pool bound to VMM per-layer arenas "
                        "(%d layers, %d chunks/layer, live_rows=%d)",
                        layer_num,
                        initial_chunks_per_layer,
                        initial_live_rows,
                    )
                except Exception as kv_exc:  # noqa: BLE001
                    logger.warning(
                        "XPool: KV pool bind_arena failed (%s); "
                        "falling back to small headroom arena",
                        kv_exc,
                    )
                    import gc

                    _partial_k: list = locals().get("k_arenas") or []
                    _partial_v: list = locals().get("v_arenas") or []
                    for _a in [*_partial_k, *_partial_v]:
                        try:
                            _a.close()
                            _a.shared_pool.release_all()
                        except Exception:  # noqa: BLE001
                            pass
                    del _partial_k, _partial_v
                    gc.collect()
                    torch.cuda.empty_cache()
                    kv_arena = None

            if kv_arena is None:
                kv_max_chunks = pages_per_fire * 4
                kv_handles = SharedHandlePool(kv_max_chunks, device=self.gpu_id)
                kv_arena = ChunkArena(
                    shared_pool=kv_handles,
                    max_chunks=kv_max_chunks,
                    mapped_chunks=0,
                    name="xpool_kv_fallback",
                )

            # Mamba rebind is skipped on the legacy path because CUDA graphs
            # have already captured the old tensor pointers.  The Python
            # SimpleMambaPool was sized to base+headroom (registry honors
            # xpool_mamba_headroom_slots) so the C++ allocator's logical
            # Grow never overflows the tensor's slot dim, and the arena
            # never moves physically (max_chunks==mapped_chunks signals
            # static state to XPoolActuator).
            logger.info(
                "XPool: mamba VMM arena ready (%d/%d chunks pre-mapped); "
                "logical-only path (tensor rebind skipped, post-graph-capture)",
                mamba_current_chunks,
                mamba_max_chunks,
            )

            self._xpool_actuator = XPoolActuator(
                kv_arena=kv_arena,
                mamba_arena=mamba_arena,
                scheduler=self.scheduler,
                kv_bytes_per_page=kv_bytes_per_page,
            )
            logger.info(
                "XPool dynamic capacity enabled (legacy post-capture path): "
                "pages_per_fire=%d mamba_chunks=%d/%d kv_bound=%s "
                "kv_bytes_per_page=%d",
                pages_per_fire,
                mamba_current_chunks,
                mamba_max_chunks,
                kv_bound,
                kv_bytes_per_page,
            )
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            logger.warning(
                "XPool actuator init failed (%s); dynamic capacity disabled", exc
            )
            self._xpool_actuator = None

    def _maybe_fire_xpool(self) -> None:
        """Drain the budgeter's pending fire plan and actuate it asynchronously.

        The actuator de-dups latched plans by op_id, so calling this on every
        budget tick is safe even though the C++ side keeps the plan latched.
        """
        if self._xpool_actuator is None:
            return
        plan = self.scheduler.pending_xpool_fire()
        if plan is not None:
            logger.debug(
                "XPool pending fire: op_id=%s direction=%s n_pages=%d",
                getattr(plan, "op_id", "?"),
                getattr(plan, "direction", "?"),
                len(getattr(plan, "page_ids", [])),
            )
            self._xpool_actuator.maybe_execute(plan)

    def _maybe_migrate_xpool(self) -> None:
        """Drain the budgeter's pending migrate plan (S2.6 Stage-0).

        Called immediately after _maybe_fire_xpool() on every budget tick.
        The actuator de-dups by op_id so calling on every tick is safe.
        """
        if self._xpool_actuator is None:
            return
        plan = getattr(self.scheduler, "pending_xpool_migrate", lambda: None)()
        if plan is not None:
            logger.debug(
                "XPool pending migrate: op_id=%s pages_needed=%d",
                getattr(plan, "op_id", "?"),
                int(getattr(plan, "pages_needed", 0)),
            )
            self._xpool_actuator.maybe_execute_migrate(plan)

    def _maybe_budget_tick(self) -> None:
        if self._budgeter_tick_s <= 0:
            self._maybe_emit_replay_metrics()
            return
        now = time.monotonic()
        if self._last_budget_tick == 0.0:
            self._last_budget_tick = now
            return
        if now - self._last_budget_tick < self._budgeter_tick_s:
            self._maybe_emit_replay_metrics()
            return
        self.scheduler.budget_tick()
        self._last_budget_tick = now
        self._maybe_fire_xpool()
        self._maybe_migrate_xpool()
        self._maybe_emit_replay_metrics()

    def _maybe_emit_replay_metrics(self) -> None:
        """Periodically append a pool-state snapshot to the replay log.

        Cheap fields only -- we call this from the hot scheduler loop, so
        any expensive aggregation (e.g. iterating all requests) is avoided.
        Returns silently when ``TOKENSPEED_REPLAY_METRICS_PATH`` is unset.
        """
        if self._replay_metrics_file is None:
            return
        now = time.monotonic()
        if self._replay_metrics_start_t == 0.0:
            self._replay_metrics_start_t = now
        if (
            self._replay_metrics_last_t > 0.0
            and now - self._replay_metrics_last_t < self._replay_metrics_interval_s
        ):
            return
        self._replay_metrics_last_t = now

        actuator = getattr(self, "_xpool_actuator", None)
        kv_arena = getattr(actuator, "kv_arena", None) if actuator else None
        mamba_arena = getattr(actuator, "mamba_arena", None) if actuator else None
        snap = {
            "t": now - self._replay_metrics_start_t,
            "queue_len": int(self.scheduler.waiting_size()),
            "decoding": int(self.scheduler.decoding_size()),
            "prefilling": int(self.scheduler.prefilling_size()),
            "retract_count": int(self.scheduler.retract_count()),
            "kv_free_pages": int(self.scheduler.available_kv_pages()),
            "kv_active_pages": int(self.scheduler.active_kv_pages()),
            "kv_mapped_pages": (
                int(getattr(kv_arena, "mapped_chunks", 0))
                if kv_arena is not None
                else 0
            ),
            "kv_headroom_pages": (
                int(getattr(kv_arena, "headroom_pages", 0))
                if kv_arena is not None
                else 0
            ),
            "mamba_mapped_chunks": (
                int(getattr(mamba_arena, "mapped_chunks", 0))
                if mamba_arena is not None
                else 0
            ),
            "fires_kv_to_mamba_total": (
                int(actuator.committed_kv_to_mamba) if actuator else 0
            ),
            "fires_mamba_to_kv_total": (
                int(actuator.committed_mamba_to_kv) if actuator else 0
            ),
            "fires_cancelled_total": (
                int(actuator.cancelled_kv_to_mamba + actuator.cancelled_mamba_to_kv)
                if actuator
                else 0
            ),
            "xpool_ewma_xfer_us_per_page": (
                float(getattr(actuator, "ewma_xfer_us_per_page", 0.0))
                if actuator
                else 0.0
            ),
            "xpool_last_fire_us": (
                float(getattr(actuator, "last_fire_us", 0.0)) if actuator else 0.0
            ),
            "xpool_last_fire_pages": (
                int(getattr(actuator, "last_fire_pages", 0)) if actuator else 0
            ),
            # S2.6: accumulated migration events since process start.
            "migrations_total": (
                int(getattr(actuator, "committed_migrate", 0)) if actuator else 0
            ),
        }
        try:
            self._replay_metrics_file.write(json.dumps(snap) + "\n")
        except Exception:  # pragma: no cover - best-effort tracing
            pass

    def _process_new_requests(self):
        recv_reqs = self.request_handler.recv_reqs()
        # Snapshot the pause state before dispatch: process_requests may flip it
        # mid-batch. If it was not blocked before but is after, a pause control
        # message was processed in this very batch — which is what makes the
        # FIFO edge below detectable (see TODO(pause-fifo)).
        pause_blocked_before = self._pause.admit_blocked
        new_req_specs, new_req_states, bootstrap_infos, abort_rids = (
            self.request_handler.process_requests(recv_reqs)
        )
        # Sweep TTL-expired abort markers every iteration. Without this
        # the map only gets cleaned inside ``mark_abort``, so a burst of
        # stale-cancel traffic followed by silence leaves the last batch
        # of entries sitting past their TTL (and potentially re-aborting
        # reused rids). Amortized O(1): expired entries are always at
        # the front of the insertion-ordered dict.
        self.output_processor.sweep_pending_aborts()
        # Abort both registered and grammar-queued requests. Without the
        # grammar_manager.mark_abort call, a request aborted mid-compile
        # would finish compiling and get admitted before being noticed.
        grammar_manager = self.request_handler.grammar_manager
        for rid in abort_rids:
            self.output_processor.mark_abort(rid)
            grammar_manager.mark_abort(rid)

        # A pause(mode="abort") cancels every in-flight request through the same
        # marker path as a client abort; they finish on their next scheduled
        # step, then the drain check resolves the pause reply.
        if self._pause.consume_abort_all():
            for rid in list(self.output_processor.rid_to_state.keys()):
                # notify_client=True: pause aborts a passive client's request,
                # so it must receive a terminating finish (unlike a client abort).
                self.output_processor.mark_abort(rid, notify_client=True)
                grammar_manager.mark_abort(rid)

        # abort/wait also cancel requests still compiling in the grammar queue:
        # they are not yet in rid_to_state or the scheduler, so the sweep above
        # and the drain check both miss them. A finished state makes the next
        # get_ready_grammar_requests pass publish them instead of admitting, so
        # they never run under post-resume weights or strand the drain.
        if self._pause.consume_cancel_grammar():
            for _, state, _ in grammar_manager.grammar_queue:
                state.set_finish_with_abort("Aborted by pause", notify_client=True)

        # On resume, flush specs buffered while paused even when no new request
        # arrives this iteration. This must run before the ``if not ready:
        # return`` guard below, which would otherwise strand buffered specs
        # until the next inbound request. Specs aborted while paused are reaped
        # in place (terminating finish + state cleanup) rather than admitted, so
        # they don't burn a scheduler slot or leak their rid — see
        # ``_reap_or_keep_buffered_spec``.
        if not self._pause.admit_blocked and self._pause.buffered_specs:
            specs = [
                spec
                for spec in self._pause.take_buffered_specs()
                if self._reap_or_keep_buffered_spec(spec)
            ]
            if specs:
                self.scheduler.submit_requests(specs)

        # Partition new requests by grammar readiness. Compile-bound requests
        # are queued in GrammarManager and admitted in a later iteration when
        # their futures resolve (see _drain_ready_grammar_requests below).
        ready = []
        for spec, state, bootstrap in zip(
            new_req_specs, new_req_states, bootstrap_infos
        ):
            # Requests pre-marked finished (e.g. invalid session ID aborted
            # in RequestHandler) skip grammar compilation entirely — we'd
            # just be wasting a compile slot on a response we're about to
            # abort anyway, and the terminal response would be delayed by
            # the compile/timeout window.
            if state.finished:
                ready.append((spec, state, bootstrap))
                continue
            if grammar_manager.process_req_with_grammar(state):
                ready.append((spec, state, bootstrap))
            else:
                grammar_manager.add_to_queue(spec, state, bootstrap)

        # Drain any previously-queued requests whose grammar just finished
        # compiling. With attn_tp > 1 this also drives the per-iter all_gather
        # that keeps grammar admission in sync across ranks.
        ready.extend(grammar_manager.get_ready_grammar_requests())

        if not ready:
            return

        admitted_specs = []
        for spec, state, bootstrap in ready:
            # Grammar-aborted (invalid grammar, timed-out compile, or missing
            # backend) requests must not enter the scheduler — they have no
            # valid grammar to mask logits with, and we don't want to spend a
            # prefill slot on a request that's already finished. Publish the
            # finish_reason directly so the client still gets a response.
            if state.finished:
                self.output_processor.publish_finished_at_admission(
                    spec.request_id, state
                )
                continue

            if isinstance(self.pd_kv_transfer, DisaggDecodeExecutor):
                state.computed_length = state.input_length
            self.output_processor.register(spec.request_id, state)
            if self.pd_kv_transfer is not None:
                self.pd_kv_transfer.register(spec.request_id, bootstrap)

            if self.memory_executor is not None:
                hashes = calc_l3_query_hashes(self.scheduler, spec.tokens)
                if hashes and len(hashes) > self.prefetch_threshold:
                    hit_pages = self.memory_executor.query_l3_pages(hashes)
                    logger.debug(
                        "[cache_op] L3 query: rid=%s hash_pages=%s hit_pages=%s threshold=%s",
                        spec.request_id,
                        len(hashes),
                        hit_pages,
                        self.prefetch_threshold,
                    )
                    spec.rolling_hashes = hashes
                    spec.storage_hit_pages = hit_pages
            admitted_specs.append(spec)

        # Pause gate: while paused, withhold new requests from the scheduler
        # (running requests keep stepping); buffered specs are flushed on resume
        # above, ahead of any newly-admitted ones, preserving FIFO order.
        #
        # TODO(pause-fifo): recv_reqs() drains the socket non-blocking, so a
        # generate request that arrived *before* a pause control message can be
        # coalesced into the same batch and reach here after the pause flipped
        # admit_blocked. Such a pre-pause request is buffered as post-pause work
        # instead of running (wait) / being aborted (abort). Correct handling
        # needs the batch processed as an ordered stream that respects the
        # control request's FIFO position. Tracked as a follow-up; until then we
        # warn when the coalescing condition is observed so it is not silent.
        if self._pause.admit_blocked:
            if admitted_specs and not pause_blocked_before:
                logger.warning(
                    "Pause engaged in the same recv batch as %d generate "
                    "request(s) (rids=%s); their FIFO order relative to the "
                    "pause is not preserved, so a pre-pause request may be "
                    "buffered as post-pause work and run only after resume. "
                    "See TODO(pause-fifo).",
                    len(admitted_specs),
                    [spec.request_id for spec in admitted_specs],
                )
            self._pause.buffer_specs(admitted_specs)
            return

        if admitted_specs:
            self.scheduler.submit_requests(admitted_specs)

    @nvtx_range("loop:commit", color="rapids")
    def _commit_forward_results(
        self,
        forward_op,
        results: ModelExecutionResult,
        on_first_token=None,
    ):
        self.request_handler.forward_ct += 1
        forward_mode = ForwardMode.from_num_extends(
            forward_op.num_extends(),
            len(forward_op.request_ids),
            has_drafter=self.server_args.speculative_algorithm is not None,
            use_target_verify=(
                self.model_executor.config.use_target_verify_forward_mode
            ),
        )
        self.request_handler._profile_batch_predicate(forward_mode)

        # post_process_forward_op calls sync() — after this, CPU tensors are ready
        is_prefill_instance = isinstance(self.pd_kv_transfer, DisaggPrefillExecutor)
        request_changes = self.output_processor.post_process_forward_op(
            forward_op,
            results,
            is_prefill_instance=is_prefill_instance,
            on_first_token=on_first_token,
        )
        # Accumulate decode stats from synced results (no GPU sync)
        if forward_op.num_extends() <= 0:
            bs = len(forward_op.request_ids)
            self.model_executor.accumulate_decode_stats(results, bs)

        return request_changes

    def _get_forward_op(self, execution_plan):
        """Return the next forward op from the given plan, or None if there is nothing to run."""
        forward_ops = execution_plan.forward
        if len(forward_ops) == 0 or len(forward_ops[0].request_ids) == 0:
            return None
        return forward_ops[0]

    def _process_pd_events(self, pd_events: list) -> list:
        processed = []
        for event in pd_events:
            processed.append(event)
            if isinstance(event, PD.SucceededEvent) and isinstance(
                self.pd_kv_transfer, DisaggPrefillExecutor
            ):
                req_id = event.request_id
                processed.extend(self.output_processor.finish_prefill_request(req_id))
            elif isinstance(event, PD.RemotePrefillDoneEvent):
                req_id = event.request_id
                bootstrap_token = event.bootstrap_token

                self.output_processor.on_remote_prefill_done(req_id, bootstrap_token)
                if isinstance(self.pd_kv_transfer, DisaggDecodeExecutor):
                    candidate_info = self.pd_kv_transfer.pop_remote_spec_candidate_ids(
                        req_id
                    )
                    if candidate_info is not None:
                        req_pool_idx, candidate_ids = candidate_info
                        self.model_executor.write_remote_spec_candidate_ids(
                            req_pool_idx, candidate_ids
                        )

        return processed

    def _get_load(self):
        """Return load metrics for the DP load balancer."""
        from tokenspeed.runtime.engine.io_struct import GetLoadReqOutput

        available = self.scheduler.available_kv_pages()
        num_total_pages = self.max_total_num_tokens // self.server_args.block_size
        num_used_pages = num_total_pages - available
        num_waiting = self.scheduler.waiting_size()
        # num_reqs: running + waiting (used by SHORTEST_QUEUE balancing)
        num_running = len(self.output_processor.rid_to_state)
        return GetLoadReqOutput(
            dp_rank=self.dp_rank,
            num_reqs=num_running + num_waiting,
            num_waiting_reqs=num_waiting,
            num_pages=num_used_pages,
        )

    def _dp_sync_and_check(self, forward_op) -> DpForwardMetadata:
        """Synchronize DP ranks with CPU-only metadata.

        All ranks call this before GPU forward work. The gathered metadata is
        used for eager token-aware collectives and for choosing a common padded
        CUDA graph shape during decode.
        """
        import torch.distributed as dist

        executes_model_forward = _forward_op_executes_model_forward(
            forward_op,
            is_disagg_decode=isinstance(self.pd_kv_transfer, DisaggDecodeExecutor),
        )
        num_tokens = sum(forward_op.input_lengths) if executes_model_forward else 0
        batch_size = len(forward_op.request_ids) if executes_model_forward else 0
        if not executes_model_forward:
            forward_mode = ForwardMode.IDLE
        else:
            forward_mode = ForwardMode.from_num_extends(
                forward_op.num_extends(),
                batch_size,
                has_drafter=self.server_args.speculative_algorithm is not None,
                use_target_verify=(
                    self.model_executor.config.use_target_verify_forward_mode
                ),
            )

        self._dp_local_info[0, 0] = num_tokens
        self._dp_local_info[0, 1] = batch_size
        self._dp_local_info[0, 2] = int(forward_mode)
        dist.all_gather_into_tensor(
            self._dp_global_info,
            self._dp_local_info,
            group=self.world_cpu_group,
        )
        global_num_tokens = self._dp_global_info[:, 0].tolist()
        global_batch_size = self._dp_global_info[:, 1].tolist()
        global_forward_mode = self._dp_global_info[:, 2].tolist()
        any_rank_has_work = max(global_num_tokens) > 0
        need_idle_forward = num_tokens == 0 and any_rank_has_work
        all_decode_or_idle = all(
            mode
            in (
                int(ForwardMode.DECODE),
                int(ForwardMode.IDLE),
                int(ForwardMode.TARGET_VERIFY),
            )
            for mode in global_forward_mode
        )
        return DpForwardMetadata(
            global_num_tokens=global_num_tokens,
            global_batch_size=global_batch_size,
            global_forward_mode=global_forward_mode,
            all_decode_or_idle=all_decode_or_idle,
            need_idle_forward=need_idle_forward,
        )

    def _get_scheduler_stats(self):
        """Query scheduler for page usage and queue depth."""
        available = self.scheduler.available_kv_pages()
        active = self.scheduler.active_kv_pages()
        num_total_pages = self.max_total_num_tokens // self.server_args.block_size
        return {
            "num_active_pages": active,
            "num_cached_pages": num_total_pages - available,
            "num_queue_reqs": self.scheduler.waiting_size(),
        }

    def _record_scheduler_iteration_metrics(
        self, stats: dict, num_iteration_tokens: int
    ) -> None:
        self.metrics.record_scheduler_iteration(
            running=len(self.output_processor.rid_to_state),
            waiting=stats["num_queue_reqs"],
            num_active_pages=stats["num_active_pages"],
            num_total_pages=self.max_total_num_tokens // self.server_args.block_size,
            num_iteration_tokens=num_iteration_tokens,
        )

    # ------------------------------------------------------------------
    # Pause / resume helpers
    # ------------------------------------------------------------------

    def _reset_caches_for_release(self) -> None:
        """Invalidate the prefix/radix cache before KV is discarded on release.

        KV pages are re-mapped + zeroed on wake, so any retained prefix entry
        would be stale. The unsafe case (prefix caching on with no reset) is
        rejected up front in ``MemoryOccupationController.handle_release`` via
        ``kv_cache_release_allowed``, so by the time we get here either a reset
        exists or prefix caching is off (nothing to invalidate).
        """
        reset = getattr(self.scheduler, "reset_prefix_cache", None)
        if callable(reset):
            reset()

    def _kv_pools(self) -> list:
        """All KV pools whose pages are tagged ``kv_cache`` — the target pool and
        the draft pool in speculative-decoding runs. Release/repair must walk the
        SAME set, so both derive it here rather than enumerating pools by hand."""
        pools = []
        for attr in ("token_to_kv_pool", "draft_token_to_kv_pool"):
            pool = getattr(self.model_executor, attr, None)
            if pool is not None:
                pools.append(pool)
        return pools

    def _kv_repair_after_wake(self) -> None:
        """Zero re-mapped KV buffers (garbage after re-map) for every KV pool,
        including the draft pool in spec-decode runs — its allocations are tagged
        ``kv_cache`` too, so a wake that skipped it would feed the draft model
        stale KV. FP8 KV scales ride with the weights region, so no scale reset
        is needed here."""
        for pool in self._kv_pools():
            if hasattr(pool, "clear_kv_buffers"):
                pool.clear_kv_buffers()

    def _paused_idle_step(self, prev_forward_op=None, prev_results=None) -> None:
        """Run one iteration under ``PAUSED_ALL`` (keep mode): no new forward
        work, but keep DP ranks in lockstep, service the drain check, and yield
        the CPU so the freeze does not busy-spin a core."""
        if prev_results is not None:
            request_changes = self._commit_forward_results(
                prev_forward_op, prev_results
            )
            advance_forward(self.scheduler, request_changes)
            self._publish_scheduler_kv_events()

        if self.has_dp:
            dp_metadata = self._dp_sync_and_check(None)
            # While memory is released the weights region is unmapped; an idle
            # forward runs the model and would read freed memory. All DP ranks
            # release together, so skipping the idle forward stays consistent
            # across ranks (the small DP sync above still runs to keep lockstep).
            if dp_metadata.need_idle_forward and not self._pause.released:
                self.model_executor.execute_idle_forward(
                    dp_metadata.global_num_tokens,
                    dp_metadata.global_batch_size,
                    dp_metadata.all_decode_or_idle,
                )

        self._pause.maybe_finish_drain(self.scheduler)
        time.sleep(_PAUSED_IDLE_SLEEP_S)

    # ------------------------------------------------------------------
    # Event loops
    # ------------------------------------------------------------------

    def event_loop(self):
        """Non-overlapping scheduler loop."""
        while True:
            self._process_new_requests()
            self._maybe_budget_tick()
            self._commit_cache_results()
            if self._pause.forward_blocked:
                self._paused_idle_step()
                continue
            execution_plan = self.scheduler.next_execution_plan()
            self._publish_scheduler_kv_events()
            self._submit_cache_ops(execution_plan)

            forward_op = self._get_forward_op(execution_plan)
            self._flush_mamba_retract_states(forward_op)

            stats = self._get_scheduler_stats()
            num_iter_tokens = (
                sum(forward_op.input_lengths) if forward_op is not None else 0
            )

            # DP sync: all ranks must participate even when idle.
            dp_metadata = None
            if self.has_dp:
                dp_metadata = self._dp_sync_and_check(forward_op)
                if dp_metadata.need_idle_forward:
                    self.model_executor.execute_idle_forward(
                        dp_metadata.global_num_tokens,
                        dp_metadata.global_batch_size,
                        dp_metadata.all_decode_or_idle,
                    )
                    self._record_scheduler_iteration_metrics(stats, num_iter_tokens)
                    continue

            request_changes = []

            if forward_op is not None:
                sampling_params_list = self._gather_sampling_params(forward_op)
                grammar_inputs = self._gather_grammar_state(forward_op)
                results, on_first_token = self._dispatch_forward(
                    forward_op,
                    sampling_params_list,
                    execution_plan,
                    dp_metadata=dp_metadata,
                    stats=stats,
                    grammar_inputs=grammar_inputs,
                )
                if results is not None:
                    request_changes.extend(
                        self._commit_forward_results(
                            forward_op, results, on_first_token
                        )
                    )

            if self.pd_kv_transfer is not None:
                pd_events = self.pd_kv_transfer.generate_events()
                request_changes.extend(self._process_pd_events(pd_events))

            if request_changes:
                advance_forward(self.scheduler, request_changes)
                self._publish_scheduler_kv_events()

            # Resolve a deferred abort/wait pause reply once in-flight work drains.
            self._pause.maybe_finish_drain(self.scheduler)

            self._record_scheduler_iteration_metrics(stats, num_iter_tokens)

    def _gather_sampling_params(self, forward_op) -> list[SamplingParams]:
        """Look up per-request SamplingParams from the output processor. The
        sampling backend does its own flip detection + RNG state management
        internally, so we only need the scalar params here."""
        return [
            self.output_processor.rid_to_state[rid].sampling_params
            for rid in forward_op.request_ids
        ]

    def _gather_grammar_state(self, forward_op) -> GrammarStepInputs | None:
        """Build ``GrammarStepInputs`` for the current batch, or ``None``.

        Returns ``None`` when no request in this batch has a grammar — the
        model_executor short-circuits then. Otherwise carries the grammars
        list + per-EXTEND-slot ``advance_mask`` (False on intermediate
        chunked-prefill chunks, since the sampled token is discarded by
        post_process and must not advance the matcher).
        """
        rid_to_state = self.output_processor.rid_to_state
        grammars = [rid_to_state[rid].grammar for rid in forward_op.request_ids]
        if not any(grammars):
            return None

        advance_mask = None
        num_extends = forward_op.num_extends()
        if num_extends > 0:
            bs = len(forward_op.request_ids)
            extend_prefix_lens = forward_op.extend_prefix_lens
            extend_input_lengths = forward_op.input_lengths[:num_extends]
            advance_mask = [True] * bs
            for i in range(num_extends):
                rid = forward_op.request_ids[i]
                # This chunk completes prefill iff it processes the final
                # token of the prompt; intermediate chunks don't.
                advance_mask[i] = (
                    extend_prefix_lens[i] + extend_input_lengths[i]
                    >= rid_to_state[rid].input_length
                )

        return GrammarStepInputs(grammars=grammars, advance_mask=advance_mask)

    def event_loop_overlap(self):
        """
        Overlapping scheduler loop: post-process the previous step's results
        while the current step's forward pass is in flight.
        """
        prev_results: ModelExecutionResult = None
        prev_forward_op = None

        while True:
            # Order this iter's default-stream writes (KVAllocator,
            # update_block_table, prefix_cache writes to req_to_page)
            # after the prev iter's forward on execution_stream that
            # reads the same tensor. Non-blocking on host.
            torch.cuda.default_stream().wait_stream(
                self.model_executor.execution_stream
            )
            self._process_new_requests()
            self._maybe_budget_tick()
            self._commit_cache_results()
            if self._pause.forward_blocked:
                # Freeze: commit any in-flight (overlapped) step — a forward
                # already on the GPU can't be un-launched — then idle.
                self._paused_idle_step(prev_forward_op, prev_results)
                prev_results = None
                prev_forward_op = None
                continue
            execution_plan = self.scheduler.next_execution_plan()
            self._publish_scheduler_kv_events()

            self._submit_cache_ops(execution_plan)

            forward_op = self._get_forward_op(execution_plan)
            self._flush_mamba_retract_states(forward_op)

            stats = self._get_scheduler_stats()
            num_iter_tokens = (
                sum(forward_op.input_lengths) if forward_op is not None else 0
            )

            grammar_inputs = None
            if forward_op is not None:
                # Gather both sampling params and grammar state BEFORE the
                # prev_results commit below — that commit can finish requests
                # and pop them from output_processor.rid_to_state, which would
                # KeyError when we look up rids that are still in the current
                # forward_op.
                sampling_params_list = self._gather_sampling_params(forward_op)
                grammar_inputs = self._gather_grammar_state(forward_op)

            # DP sync: all ranks must participate even when idle.
            dp_metadata = None
            if self.has_dp:
                dp_metadata = self._dp_sync_and_check(forward_op)
                if dp_metadata.need_idle_forward:
                    if prev_results is not None:
                        request_changes = self._commit_forward_results(
                            prev_forward_op, prev_results
                        )
                        advance_forward(self.scheduler, request_changes)
                        self._publish_scheduler_kv_events()
                        prev_results = None
                        prev_forward_op = None
                    self.model_executor.execute_idle_forward(
                        dp_metadata.global_num_tokens,
                        dp_metadata.global_batch_size,
                        dp_metadata.all_decode_or_idle,
                    )
                    self._record_scheduler_iteration_metrics(stats, num_iter_tokens)
                    continue

            # ---- dispatch current forward first (async GPU launch) ----
            # Issue curr's forward before committing prev so the GPU runs curr
            # while the CPU syncs/post-processes prev. Committing prev first
            # would block the CPU on prev's copy_event and leave the GPU idle
            # until dispatch — visible as a gap between forwards in the trace.
            #
            # Eager grammar exception: setup_grammar_step reads each matcher's
            # current state to fill the bitmask. Under the overlap pattern the
            # matcher hasn't been advanced yet by prev's accept_token (commit
            # below), so the fill would use a one-step-stale state and let the
            # model sample a token the matcher then rejects. Capturable
            # grammar dodges this with an in-graph hostfunc that advances
            # before fill; eager has no equivalent, so we commit prev first
            # whenever this batch carries grammars. Costs the dispatch/commit
            # overlap for grammar batches but is correct.
            request_changes = []
            curr_has_grammar = grammar_inputs is not None
            eager_grammar_needs_advance = (
                curr_has_grammar
                and prev_results is not None
                and self.model_executor.eager_grammar_buffers is not None
            )
            if eager_grammar_needs_advance:
                request_changes.extend(
                    self._commit_forward_results(prev_forward_op, prev_results)
                )
                prev_results = None
                prev_forward_op = None

            curr_results = None
            if forward_op is not None:
                curr_results, _ = self._dispatch_forward(
                    forward_op,
                    sampling_params_list,
                    execution_plan,
                    dp_metadata=dp_metadata,
                    stats=stats,
                    grammar_inputs=grammar_inputs,
                )

            # ---- post-process previous step (overlapped with current forward) ----
            if prev_results is not None:
                request_changes.extend(
                    self._commit_forward_results(prev_forward_op, prev_results)
                )

            # ---- collect PD events ----
            if self.pd_kv_transfer is not None:
                pd_events = self.pd_kv_transfer.generate_events()
                request_changes.extend(self._process_pd_events(pd_events))

            if request_changes:
                advance_forward(self.scheduler, request_changes)
                self._publish_scheduler_kv_events()

            # Resolve a deferred abort/wait pause reply once in-flight work drains.
            self._pause.maybe_finish_drain(self.scheduler)

            self._record_scheduler_iteration_metrics(stats, num_iter_tokens)

            prev_results = curr_results
            prev_forward_op = forward_op


def run_event_loop(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
):
    mapping = server_args.mapping
    gpu_id = mapping.rank % mapping.nprocs_per_node + server_args.base_gpu_id
    attn_tp_rank = mapping.attn.tp_rank
    dp_rank = mapping.attn.dp_rank
    global_rank = mapping.rank

    setproctitle.setproctitle(f"tokenspeed::scheduler_{dp_rank}")
    faulthandler.enable()
    parent_process = psutil.Process().parent()
    register_usr_signal()

    prefix = f" ATTN TP RANK {attn_tp_rank}"
    configure_logger(server_args, prefix=prefix)

    try:
        event_loop = EventLoop(
            server_args,
            port_args,
            gpu_id,
            attn_tp_rank,
            dp_rank,
            global_rank,
        )
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": event_loop.max_total_num_tokens,
                "max_req_input_len": event_loop.max_req_input_len,
                "max_num_seqs": server_args.max_num_seqs,
                "chunked_prefill_size": server_args.chunked_prefill_size,
                "max_model_len": event_loop.model_config.context_len,
            }
        )

        use_overlap = should_use_overlap_schedule(
            disable_overlap_schedule=server_args.disable_overlap_schedule,
            disaggregation_mode=server_args.disaggregation_mode,
            speculative_algorithm=server_args.speculative_algorithm,
            paged_cache_groups=getattr(event_loop, "_paged_cache_groups", ()),
        )
        if use_overlap:
            event_loop.event_loop_overlap()
        else:
            event_loop.event_loop()

    except Exception:
        traceback = get_exception_traceback()
        logger.error("Scheduler hit an exception: %s", traceback)
        parent_process.send_signal(signal.SIGUSR1)
