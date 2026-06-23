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

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "resource/types.h"
#include "scheduler/types.h"
#include "scheduler/request.h"
#include "scheduler/execution_plan.h"
#include "scheduler/execution_event.h"
#include "scheduler/kv_cache_events.h"

#include "resource/allocator/page_allocator.h"
#include "resource/allocator/paged_cache_group.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"
#include "resource/allocator/req_pool_allocator.h"
#include "resource/allocator/mamba_chunk_allocator.h"
#include "resource/allocator/mamba_host_allocator.h"
#include "resource/hybrid_prefix_cache/hybrid_prefix_cache.h"
#include "budgeter/budget_agent.h"

#include "fsm/forward_events.h"
#include "fsm/cache_events.h"
#include "fsm/pd_events.h"
namespace tokenspeed {

class Scheduler {
public:
    explicit Scheduler(SchedulerConfig config);

    void SubmitRequests(const std::vector<RequestSpec>& request_specs);
    std::vector<std::string> CalcRollingHash(const std::vector<std::int32_t>& input_tokens, bool apply_match = false);

    ExecutionPlan NextExecutionPlan();

    void Advance(const ExecutionEvent& event);
    std::vector<KvCacheEvent> DrainKvEvents();

    std::size_t WaitingSize() const;
    std::size_t DecodingSize() const;
    std::size_t RetractedSize() const;
    std::size_t AvailableKvPages() const;
    std::size_t ActiveKvPages() const;
    std::size_t PrefillSize() const;
    std::int32_t GetRequestTokenSize(const std::string& id) const;
    std::vector<std::string> PagedCacheGroupIds() const;
    std::int32_t PagedCacheGroupTotalPages(const std::string& group_id) const;
    std::int32_t PagedCacheGroupAvailablePages(const std::string& group_id) const;
    std::int64_t PagedCacheGroupFailedAllocCount(const std::string& group_id) const;
    std::vector<std::int32_t> GetRequestPagedCachePageIds(const std::string& request_id,
                                                          const std::string& group_id) const;
    // Compact-view base logical-page offset; 0 for full-history / unseen.
    std::int32_t GetRequestPagedCacheBaseLogicalPage(const std::string& request_id, const std::string& group_id) const;

    void BudgetTick();
    std::optional<XPoolFirePlan> PendingXPoolFire() const;

    // S2.6 migration plan: latched when the admitter selects kCrossMigrate.
    // The Python actuator reads this, calls BestMigrateCandidate() to find a
    // victim, retracts it, and then calls ApplyXPoolMigrate() to commit.
    std::optional<XPoolMigratePlan> PendingXPoolMigrate() const;
    void ApplyXPoolMigrate(const XPoolMigratePlan& plan);
    void CancelXPoolMigrate();

    // Returns the request ID of the best migration candidate (the Decoding or
    // PrefillDone request with the most active KV pages).  Empty string when
    // no suitable candidate exists.
    std::string BestMigrateCandidate() const;

    // Apply the capacity changes described by a fire plan after the Python
    // actuator has completed the corresponding cuMemMap / cuMemUnmap ops.
    // Calls Grow/Shrink on both KV and mamba allocators and clears the
    // pending fire latch so the budgeter can issue new plans.
    void ApplyXPoolFire(const XPoolFirePlan& plan);

    // Shrink-and-drain helpers for kv_to_mamba direction.
    //
    // PrepareKvToMambaFire() caps the tail KV pages BEFORE physical unmap so
    // no new allocations land on pages that are about to be unmapped.  Must be
    // called from the Python actuator before cuMemUnmap; ApplyXPoolFire() will
    // skip the Shrink if it was already done here.
    void PrepareKvToMambaFire(std::int32_t n_kv_pages);

    // True while any capped KV page is still held by an in-flight request.
    // The Python actuator polls this until it returns false before unmapping.
    bool HasCappedKvInflight() const;

    // Symmetric shrink-and-drain helpers for mamba_to_kv direction.
    //
    // PrepareMambaToKvFire() caps the tail mamba slots BEFORE physical unmap so
    // no new allocations land on slots that are about to be transferred to KV.
    // HasCappedMambaInflight() polls until all capped slots are freed.
    void PrepareMambaToKvFire(std::int32_t n_mamba_slots);
    bool HasCappedMambaInflight() const;

    // Clears the pending fire latch WITHOUT updating allocator capacities.
    // Call this when the Python actuator decides to skip the physical VMM step
    // (e.g. arena headroom is exhausted) so the budgeter can emit new plans.
    void CancelXPoolFire();

    // Capacity observability helpers (for Python-side monitoring).
    std::int32_t MappedKvPages() const;
    std::int32_t AvailableMambaSlots() const;
    std::int32_t MappedMambaSlots() const;

    // Dynamic admission cap (S2.7).  Returns the current effective
    // max_batch_size (== config_.max_batch_size), and lets callers
    // shrink it.  The setter clamps to [1, original_max_batch_size_]
    // because the ReqPoolAllocator is sized for the boot-time value
    // and cannot grow beyond it.  Used by BudgetTick when
    // enable_dynamic_admission_cap is on, and exposed via pybind so
    // operators can drive the cap directly during ablation studies.
    std::int32_t MaxBatchSize() const { return config_.max_batch_size; }
    void SetMaxBatchSize(std::int32_t new_cap);

private:
    // Second element is LoadBackOperation list (normal path) or WriteBackOperation list (retract triggered).
    std::tuple<std::vector<ForwardOperation>,
               std::variant<std::vector<LoadBackOperation>, std::vector<WriteBackOperation>>>
    newForwardOperation(std::vector<Request*> candidates);
    std::vector<WriteBackOperation> newWriteBackOperation(
        std::unordered_map<std::string, std::unique_ptr<Request>>& requests);
    std::optional<WriteBackOperation> newRetractOperation(Request* retract_request);

    PrefillOperation applyEventAndGenerateOp(Request* request, fsm::SchedulePrefillFirstChunkEvent event);
    PrefillOperation applyEventAndGenerateOp(Request* request, fsm::SchedulePrefillEvent event);
    DecodeOperation applyEventAndGenerateOp(Request* request, fsm::ScheduleDecodeEvent event);
    DecodeOperation applyEventAndGenerateOp(Request* request, fsm::ScheduleDecodeFromRetractedEvent event);
    std::optional<WriteBackOperation> applyEventAndGenerateOp(Request* request, fsm::ScheduleRetractEvent event);
    PrefetchOperation applyEventAndGenerateOp(Request* request, fsm::SchedulePrefetchEvent event);

    std::optional<fsm::SchedulePrefetchEvent> schedulePrefetch(Request* request, const MatchResult& match);

    std::optional<fsm::SchedulePrefillFirstChunkEvent> schedulePrefillFirstChunk(
        Request* request, std::int32_t remaining, std::int32_t reserve_num_tokens_in_next_schedule_event,
        bool disable_l2_cache, std::map<std::string, std::int32_t>& simulated_free);
    std::optional<fsm::SchedulePrefillEvent> schedulePrefill(Request* request, std::int32_t remaining,
                                                             std::int32_t reserve_num_tokens_in_next_schedule_event,
                                                             std::map<std::string, std::int32_t>& simulated_free);
    std::optional<fsm::ScheduleDecodeEvent> scheduleDecode(Request* request,
                                                           std::map<std::string, std::int32_t>& simulated_free);
    std::optional<fsm::ScheduleDecodeFromRetractedEvent> scheduleDecodeFromRetracted(
        Request* request, std::map<std::string, std::int32_t>& simulated_free);
    std::optional<fsm::ScheduleRetractEvent> scheduleRetract(Request* request);

    void check_device_mem();

private:
    void handleEvent(const cache::PrefetchDone& event);
    void handleEvent(const cache::WriteBackDone& event);
    void handleEvent(const pd::BootstrappedEvent& event);
    void handleEvent(const pd::FailedEvent& event);
    void handleEvent(const pd::SucceededEvent& event);
    void handleEvent(const pd::RemotePrefillDoneEvent& event);
    void handleEvent(const forward::ExtendResult& event);
    void handleEvent(const forward::Abort& event);
    void handleEvent(const forward::Finish& event);
    void handleEvent(const forward::UpdateReserveNumTokens& event);

    PoolSnapshot MakePoolSnapshot() const;

private:
    Request* find_request(std::string rid) {
        auto it = requests_.find(rid);
        return it != requests_.end() ? it->second.get() : nullptr;
    }

private:
    // S2.7: remember the boot-time max_batch_size so SetMaxBatchSize()
    // can clamp dynamic-cap reductions back up safely.  Initialised from
    // config_.max_batch_size in the Scheduler constructor.
    std::int32_t original_max_batch_size_{};
    SchedulerConfig config_;
    // Tracks how many KV/mamba pages have been pre-shrunk via the Prepare*
    // helpers so that ApplyXPoolFire does not double-shrink.
    std::int32_t kv_pre_shrunk_pages_{0};
    std::int32_t mamba_pre_shrunk_slots_{0};

private:
    PageAllocator device_allocator_;
    PageAllocator host_allocator_;
    std::optional<MambaChunkAllocator> mamba_allocator_{};
    std::optional<MambaHostAllocator> mamba_host_allocator_{};
    KVPrefixCache kv_prefix_cache_;
    ReqPoolAllocator req_pool_allocator_;
    std::optional<HybridPrefixCache> hybrid_prefix_cache_{};
    std::optional<BudgetAgent> budget_agent_{};

private:
    std::unordered_map<std::string, std::unique_ptr<Request>> requests_;
    std::unordered_map<cache_op_id, CacheOpSpec> cache_op_tracker_;
    std::vector<KvCacheEvent> kv_events_;
    // Stats
    SchedulerStats stats_;
};

}  // namespace tokenspeed
