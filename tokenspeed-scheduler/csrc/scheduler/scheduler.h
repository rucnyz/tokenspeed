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

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
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

#if TOKENSPEED_FLAT_KVCACHE
#include <unordered_set>

#include "cache/block_pool.h"
#include "cache/kv_cache_coordinator.h"
#include "cache/forward_cache_ops.h"
#endif
namespace tokenspeed {

class Scheduler {
public:
    explicit Scheduler(SchedulerConfig config);

    void SubmitRequests(const std::vector<RequestSpec>& request_specs);
    std::vector<std::string> CalcRollingHash(const std::vector<std::int32_t>& input_tokens, bool apply_match = false);

    ExecutionPlan NextExecutionPlan();

    // Overlap-scheduling retract safety (2026-07-18).
    //
    // The Python overlap event loop dispatches forward batch N, then plans
    // batch N+1 (this NextExecutionPlan call) BEFORE committing batch N's
    // results. Any retract decided during that planning races batch N's
    // still-running kernels if the victim is part of batch N: applyRetract
    // frees the victim's mamba working slot (reallocated to another request
    // while the kernel still writes its recurrent state) and WriteBackDone
    // later drops its device tree ref (pages evicted/reused, or with XPool
    // fires physically cuMemUnmap-ed, while the kernel still reads its KV
    // prefix). Observed repeatedly as CUDA illegal-memory-access seconds
    // after retracting an actively-decoding victim (long_horizon sys reps,
    // 2026-07-18).
    //
    // The event loop therefore reports the still-uncommitted batch's request
    // ids immediately before each NextExecutionPlan() call, and every
    // retract victim-selection path (S2.1 OOM fallback, S2.5-followup-2
    // proactive drain retract, S2.6 migrate candidate) skips these requests.
    // This defers a retract by at most one iteration: if OOM leaves this
    // plan with no forward op, nothing new gets dispatched, batch N commits
    // at the end of the current iteration, and the NEXT planning call sees
    // an empty in-flight set so the victim can be retracted with no kernel
    // in flight -- the same ordering a non-overlap loop provides.
    //
    // Args:
    //   request_ids: request ids in the dispatched-but-uncommitted forward
    //     batch; pass an empty vector when nothing is in flight.
    void SetInflightRequests(const std::vector<std::string>& request_ids);

    // True if the request is in the dispatched-but-uncommitted forward batch
    // reported by the latest SetInflightRequests() call.
    bool IsRequestInflight(const std::string& request_id) const;

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
#if TOKENSPEED_FLAT_KVCACHE
    // Free pages in the flat shared BlockPool; int32 twin of AvailableKvPages() for C++ tests.
    std::int32_t FlatPoolFreeBlocks() const { return block_pool_.NumFreeBlocks(); }
    std::int32_t FlatHostPoolCachedBlocks() const { return flat_host_pool_.NumCachedBlocks(); }
    std::int32_t FlatHostPoolFreeBlocks() const { return flat_host_pool_.NumFreeBlocks(); }
    std::int32_t FlatHostPoolPinnedBlocks() const { return flat_host_pool_.NumPinnedCachedBlocks(); }
#endif

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

    // Count of capped KV pages still held by in-flight requests (0 when
    // drained).  Exposes the underlying signal behind HasCappedKvInflight()
    // so the Python actuator's _wait_drain can detect *no-progress* (count
    // stopped decreasing) and abandon a pinned fire in sub-second time,
    // instead of holding the prepare tail-cap for the full drain timeout.
    std::int32_t CappedKvInflightCount() const;

    // Symmetric shrink-and-drain helpers for mamba_to_kv direction.
    //
    // PrepareMambaToKvFire() caps the tail mamba slots BEFORE physical unmap so
    // no new allocations land on slots that are about to be transferred to KV.
    // HasCappedMambaInflight() polls until all capped slots are freed.
    void PrepareMambaToKvFire(std::int32_t n_mamba_slots);
    bool HasCappedMambaInflight() const;

    // Count of capped mamba slots still held by in-flight requests (0 when
    // drained).  See CappedKvInflightCount() for the rationale.
    std::int32_t CappedMambaInflightCount() const;

    // Record a wall-clock backoff for proactive retract after scheduleRetract
    // page-accounting mismatch on this request (S2.5-followup-2).
    void RecordXPoolProactiveRetractBackoff(const std::string& request_id);

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
    // S2.5-followup (proactive retract): when capped KV/mamba pages are still
    // held by active decoders after Prepare*Fire, retract the worst offender so
    // drain_sync can finish sooner than waiting for natural decode completion.
    std::vector<WriteBackOperation> newXPoolCappedDrainRetractOperations();
    std::optional<WriteBackOperation> newRetractOperation(Request* retract_request);

    PrefillOperation applyEventAndGenerateOp(Request* request, fsm::SchedulePrefillFirstChunkEvent event,
                                             std::vector<LoadBackOperation>& loadback_ops);
    PrefillOperation applyEventAndGenerateOp(Request* request, fsm::SchedulePrefillEvent event);
    DecodeOperation applyEventAndGenerateOp(Request* request, fsm::ScheduleDecodeEvent event);
    DecodeOperation applyEventAndGenerateOp(Request* request, fsm::ScheduleDecodeFromRetractedEvent event);
    std::optional<WriteBackOperation> applyEventAndGenerateOp(Request* request, fsm::ScheduleRetractEvent event);
    PrefetchOperation applyEventAndGenerateOp(Request* request, fsm::SchedulePrefetchEvent event);

    std::optional<fsm::SchedulePrefetchEvent> schedulePrefetch(Request* request, const MatchResult& match);

    std::optional<fsm::SchedulePrefillFirstChunkEvent> schedulePrefillFirstChunk(
        Request* request, std::int32_t remaining, std::int32_t decode_input_tokens, bool disable_l2_cache,
        std::map<std::string, std::int32_t>& simulated_free);
    std::optional<fsm::SchedulePrefillEvent> schedulePrefill(Request* request, std::int32_t remaining,
                                                             std::int32_t reserve_num_tokens_in_next_schedule_event,
                                                             std::map<std::string, std::int32_t>& simulated_free);
    std::optional<fsm::ScheduleDecodeEvent> scheduleDecode(Request* request,
                                                           std::map<std::string, std::int32_t>& simulated_free);
    std::optional<fsm::ScheduleDecodeFromRetractedEvent> scheduleDecodeFromRetracted(
        Request* request, std::map<std::string, std::int32_t>& simulated_free);
    std::optional<fsm::ScheduleRetractEvent> scheduleRetract(Request* request);

#if TOKENSPEED_FLAT_KVCACHE
    // One hash pass at admission: the device match, the read-only host-tier match above its
    // boundary, and the extension's hash slice (registration form).
    struct FlatAdmissionMatch {
        CoordinatorMatch device;
        CoordinatorMatch host;
        std::vector<std::string> ext_hashes;
    };
    FlatAdmissionMatch matchFlatPrefixAtAdmission(Request* request);
    std::optional<std::int32_t> flatAdmitFirstChunk(Request* request, const CoordinatorMatch& hit,
                                                    std::int32_t ext_real_pages, std::int32_t chunk_tokens,
                                                    std::int32_t decode_reserve_tokens) const;
    std::optional<std::int32_t> flatAdmitPrefillChunk(Request* request, std::int32_t chunk_tokens,
                                                      std::int32_t decode_reserve_tokens,
                                                      std::int32_t num_computed_tokens) const;
    bool flatAdmitDecode(Request* request) const;
    bool flatPoolWedged(const std::vector<Request*>& candidates) const;
    void resolveFlatStarvation(const std::vector<Request*>& candidates, bool made_progress);
#endif

    void check_device_mem();

private:
    void handleEvent(const cache::PrefetchDone& event);
    void handleEvent(const cache::WriteBackDone& event);
    void handleEvent(const cache::LoadBackDone& event);
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

    // Group-id list for flat KV-cache ops; empty span on the radix path so call sites stay #if-free.
    std::span<const std::string> FlatGroupIds() const {
#if TOKENSPEED_FLAT_KVCACHE
        return flat_group_ids_;
#else
        return {};
#endif
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

    // S2.5-followup-2: proactive retract runs on every scheduler tick.
    // After a scheduleRetract page-accounting mismatch (or a failed attempt),
    // suppress re-selecting that request until wall-clock backoff expires.
    // Duration reuses xpool_reverse_cooldown_s (same knob as fire direction
    // thrash guard). Entries are pruned when capped inflight clears.
    std::unordered_map<std::string, std::chrono::steady_clock::time_point>
        xpool_proactive_retract_backoff_until_;

    bool IsXPoolProactiveRetractInBackoff(const std::string& request_id) const;
    void PruneExpiredXPoolProactiveRetractBackoff();

    // See SetInflightRequests(). Requests in this set must not be selected
    // as retract/migrate victims during the current planning call.
    std::unordered_set<std::string> inflight_request_ids_;

private:
    PageAllocator device_allocator_;
    PageAllocator host_allocator_;
    std::optional<MambaChunkAllocator> mamba_allocator_{};
    std::optional<MambaHostAllocator> mamba_host_allocator_{};
    KVPrefixCache kv_prefix_cache_;
    ReqPoolAllocator req_pool_allocator_;
    std::optional<HybridPrefixCache> hybrid_prefix_cache_{};
    std::optional<BudgetAgent> budget_agent_{};

#if TOKENSPEED_FLAT_KVCACHE
    BlockPool block_pool_;
    // Host tier = a second BlockPool, isomorphic to the device pool (block 0 is the null
    // placeholder there too); the two differ only in which memory the ids index.
    BlockPool flat_host_pool_;
    KvCacheCoordinator coordinator_;
    std::vector<std::string> flat_group_ids_;  // group_id per cache group, index-aligned to coordinator groups
    // ExtendResults the executor still owes per request (erased on Finish/Abort/PD-success); non-empty means
    // an in-flight forward can still free pool pages, which flatPoolWedged keys off.
    std::unordered_map<std::string, std::int32_t> pending_forward_results_;
    // Reserve ledger: decode pages promised at admission but Acquired only at PrefillDone->Decoding; until
    // then they sit in the free count, so every flat gate subtracts OTHER requests' entries.
    std::unordered_map<std::string, std::int32_t> flat_reserved_pages_;
    // Flat retract requires TWO consecutive starved rounds (an in-flight Finish fakes one)
    // before releasing a victim; see resolveFlatStarvation.
    std::int32_t flat_starved_rounds_{0};
    // Requests terminalized as flat OOM (pool wedged by unretractable mid-prefill holders, no
    // retract victim); drained into the plan being built for the client layer to fail them.
    std::vector<std::string> flat_oom_request_ids_;

    struct FlatStoreTicket {
        std::string key;
        BlockRef device_block;  // source page, pinned under the D2H copy
        BlockRef host_block;    // destination page, unhashed until WriteBackDone publishes it
    };
    // In-flight D2H stores. The host pool is transaction-blind like the device pool, so the
    // key-dedupe index lives here, paired with the op ledger: Add/Retire are the only mutation
    // points, so keys_ always equals the union of in-flight ticket keys.
    class FlatStoreLedger {
    public:
        void Add(cache_op_id id, std::vector<FlatStoreTicket> tickets) {
            for (const FlatStoreTicket& t : tickets) {
                keys_.insert(t.key);
            }
            const bool inserted = ops_.emplace(id, std::move(tickets)).second;
            _assert(inserted, "duplicate flat store op id");
        }
        // Empty result: unknown op (the radix WriteBackDone path owns it).
        std::vector<FlatStoreTicket> Retire(cache_op_id id) {
            auto it = ops_.find(id);
            if (it == ops_.end()) {
                return {};
            }
            for (const FlatStoreTicket& t : it->second) {
                keys_.erase(t.key);
            }
            std::vector<FlatStoreTicket> tickets = std::move(it->second);
            ops_.erase(it);
            return tickets;
        }
        bool InFlight(const std::string& key) const { return keys_.contains(key); }
        bool Empty() const { return ops_.empty(); }

    private:
        std::unordered_map<cache_op_id, std::vector<FlatStoreTicket>> ops_;
        std::unordered_set<std::string> keys_;
    };
    FlatStoreLedger flat_store_ops_;

    struct FlatLoadTicket {
        std::vector<BlockRef> host_pins;
        std::vector<BlockRef> device_blocks;
    };
    // In-flight H2D loads: op_id -> the pinned source host pages plus the pinned destination
    // device pages (a freed destination must not be recycled under the copy); LoadBackDone drops both.
    std::unordered_map<cache_op_id, FlatLoadTicket> flat_load_ops_;

    // Sum excluding request_id: a request consuming its own reservation must not be gated by it.
    std::int32_t flatReservedPagesExcept(const std::string& request_id) const {
        std::int32_t total = 0;
        for (const auto& [id, pages] : flat_reserved_pages_) {
            if (id != request_id) {
                total += pages;
            }
        }
        return total;
    }

    // Pool budget the flat gates charge against: free blocks minus other requests' decode reservations.
    std::int32_t flatFreeBudget(const std::string& request_id) const {
        return block_pool_.NumFreeBlocks() - flatReservedPagesExcept(request_id);
    }
#endif

private:
    std::unordered_map<std::string, std::unique_ptr<Request>> requests_;
    std::unordered_map<cache_op_id, CacheOpSpec> cache_op_tracker_;
    std::vector<KvCacheEvent> kv_events_;
    SchedulerStats stats_;
};

}  // namespace tokenspeed
