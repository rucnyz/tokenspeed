// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <optional>
#include <unordered_map>
#include <variant>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>

#include "fsm/forward_events.h"
#include "resource/allocator/paged_cache_group.h"
#include "resource/eviction_config.h"
#include "resource/types.h"
#include "scheduler/operations/inc.h"

namespace tokenspeed {

class TreeNode;

enum class DisaggregationMode {
    kNone,
    kPrefill,
    kDecode,
};
// `PagedCacheGroupFamily` is defined in
// resource/allocator/paged_cache_group.h (transitively included above).

template <ResourceType>
class NodeRef;
using HostNodeRef = NodeRef<ResourceType::Host>;
using DeviceNodeRef = NodeRef<ResourceType::Device>;

struct SchedulerStats {
    std::int64_t total_batches = 0;
    std::int64_t mixed_batches = 0;
    std::int64_t retract_count = 0;
    std::int64_t abort_count = 0;
    std::int64_t schedule_latency_count = 0;
    std::int64_t schedule_latency_sum_us = 0;
    std::int64_t schedule_latency_max_us = 0;
    std::int64_t prefix_cache_hit_tokens = 0;
    std::int64_t prefix_cache_req_tokens = 0;

    std::int64_t pending_queue_size = 0;
    std::int64_t plan_queue_size = 0;
    std::int64_t event_queue_size = 0;
    std::int64_t active_requests = 0;
};

// Opt-in spec for the paged-cache prefix-cache adjunct. Unset means paged-cache
// groups are transport-only (no snapshot chain, no prefix-cache reuse).
struct PrefixCacheAdjunctSpec {
    std::vector<std::string> required_groups{};
};

struct SchedulerConfig {
    std::int32_t page_size{};
    struct {
        std::int32_t total_pages{};
    } host_allocator;

    struct {
        std::int32_t total_pages{};
    } device_allocator;

    std::vector<PagedCacheGroupConfig> paged_cache_groups{};

    // Unset means paged-cache groups are transport-only.
    std::optional<PrefixCacheAdjunctSpec> prefix_cache_adjunct{};

    std::int32_t max_scheduled_tokens{};
    std::int32_t max_batch_size{};
    std::int32_t decode_input_tokens{1};
    bool disable_l2_cache{false};
    bool enable_l3_storage{false};
    std::int32_t prefetch_threshold{4};  // num pages
    bool enable_kv_cache_events{false};
    bool enable_mixed_prefill_decode{false};

    std::int32_t num_pages_reserved_for_retracted_or_running{};
    Role role{Role::kFused};

    bool disable_prefix_cache{false};
    bool enable_mamba{false};
    std::int32_t mamba_cache_chunk_size{64};
    std::int32_t mamba_pool_total_chunks{0};
    bool enable_mamba_l2{false};
    std::int32_t mamba_l2_host_slots{0};

    // Intra-pool eviction policy (LRU baseline or LPB loss-per-byte).
    std::string eviction_policy{"lru"};
    double lpb_window_s{60.0};
    std::int32_t lpb_hit_deque_maxlen{4096};
    double c_kv_alpha{1.02e-7};
    double c_kv_beta{0.0246};
    double c_kv_gamma{5.97};
    double c_m{0.0};
    std::int64_t kv_bytes_per_page{0};
    std::int64_t mamba_bytes_per_slot{0};

    // Inter-pool HiMA (XPool) controls.
    bool enable_budgeter{false};
    bool enable_admitter{false};
    bool enable_xpool_dynamic_capacity{false};
    double budgeter_tick_s{1.0};
    std::int32_t budgeter_pages_per_fire{64};
    double xpool_ewma_tau_s{1.0};
    double xpool_nb_margin{0.05};
    std::int32_t xpool_mamba_floor_slots{32};
    // Reverse-direction cooldown (S2.2, HiMA Phase 3). After a fire physically
    // commits in direction D (Scheduler::ApplyXPoolFire), suppress any fire in
    // the OPPOSITE direction for this many seconds. Same-direction fires are
    // never gated (they accumulate capacity transfer). Set to 0.0 to disable.
    //
    // Rationale: each fire is ~hundreds of microseconds of cuMemMap/Unmap +
    // a drain wait; without a cooldown, oscillating pressure (e.g. a burst
    // of short requests landing immediately after a fire) can cause the
    // budgeter to flip direction within the same tick window, doing twice
    // the VMM work for zero net capacity change.
    double xpool_reverse_cooldown_s{2.0};
    // Saturation gate (S2.1, HiMA Phase 3). When BOTH pools' EWMA-smoothed
    // pressure stay below this threshold, the budgeter:
    //   (a) updates EWMA but skips the fire-decision branches in Tick(), and
    //   (b) skips the per-request admit check in OnRequestArrival().
    // The intent: when neither pool is anywhere near full, inter-pool fires
    // and per-request admit gating cannot pay back their own CPU/lock
    // overhead. Default 0.5 matches the SGLang reference's `xpool_*_low`
    // range; set to 0.0 to disable (always run the full decision path).
    double xpool_saturation_low{0.5};
    double xpool_xfer_us_per_page{70.0};
    double xpool_queue_wait_us{1000.0};

    // PressureAdapter weights (S2.3, HiMA Phase 3). The budgeter's
    // direction decision is normally based on pure occupancy:
    //   ewma_pressure_kv vs ewma_pressure_mamba.
    // The PressureAdapter blends in forward-looking system signals so the
    // budgeter reacts to queue buildup and retraction events *before* the
    // raw utilisation EWMA catches up.  Each weight is the maximum
    // pressure increment its signal can contribute:
    //   adj_pressure_kv    = ewma_pressure_kv
    //                      + w_queue   * clamp01(queue_len   / queue_ref)
    //                      + w_retract * clamp01(retracted   / retract_ref)
    //   adj_pressure_mamba = ewma_pressure_mamba
    //                      + w_paused  * clamp01(paused      / paused_ref)
    // Reference counts default to max_batch_size/2 (queue) and
    // max_batch_size/4 (retract / paused); set to 0 to fall back to
    // max_batch_size when initialising.  All weights default to 0 which
    // preserves the pre-S2.3 behaviour.  Users can opt-in by setting
    // the corresponding weight via ServerArgs.
    double xpool_w_queue{0.0};
    double xpool_w_retract{0.0};
    double xpool_w_paused{0.0};
    std::int32_t xpool_queue_ref{0};
    std::int32_t xpool_retract_ref{0};
    std::int32_t xpool_paused_ref{0};

    // Initial live capacity for dynamic mode. The VA window (total_pages /
    // mamba_pool_total_chunks) should be larger than these values to leave room
    // for inter-pool transfers in both directions.  0 means "fill to maximum"
    // (= total_pages - 1 for KV, = mamba_pool_total_chunks for mamba).
    std::int32_t xpool_initial_kv_pages{0};
    std::int32_t xpool_initial_mamba_slots{0};

    // Dynamic admission cap (S2.7, HiMA Phase 3).  When enabled, each
    // BudgetTick clamps the effective max_batch_size to the current
    // mamba_total_slots, so the scheduler stops admitting new decoders
    // once kv_to_mamba fires have shrunk the Mamba pool below the
    // boot-time batch cap.  The cap is restored automatically when
    // mamba_to_kv fires return slots.  The cap never exceeds the
    // boot-time max_batch_size (the ReqPoolAllocator is sized for
    // that value and cannot grow).  Defaults to false to preserve the
    // pre-S2.7 admission behaviour.
    bool enable_dynamic_admission_cap{false};
};

}  // namespace tokenspeed
