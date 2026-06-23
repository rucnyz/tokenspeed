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

#include "scheduler/scheduler.h"

#include "budgeter/budget_agent.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <spdlog/spdlog.h>

#include "fsm/cache_states.h"
#include "fsm/forward_events.h"
#include "fsm/forward_states.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"
#include "resource/eviction_config.h"
#include "resource/radix_tree/radix_tree.h"
#include "resource/radix_tree/tree_node.h"
#include "scheduler/execution_event.h"
#include "scheduler/operations/cache.h"
#include "scheduler/page_hasher.h"
#include "scheduler/request.h"
#include "scheduler/request_spec.h"
#include "scheduler/types.h"

namespace tokenspeed {

namespace {

EvictionConfig MakeEvictionConfig(const SchedulerConfig& config) {
    EvictionConfig out;
    out.policy = EvictionConfig::ParsePolicy(config.eviction_policy);
    out.lpb_window_s = config.lpb_window_s;
    out.lpb_hit_deque_maxlen = config.lpb_hit_deque_maxlen;
    out.c_kv_alpha = config.c_kv_alpha;
    out.c_kv_beta = config.c_kv_beta;
    out.c_kv_gamma = config.c_kv_gamma;
    out.c_m = config.c_m;
    out.kv_bytes_per_page = config.kv_bytes_per_page;
    out.mamba_bytes_per_slot = config.mamba_bytes_per_slot;
    out.mamba_cache_chunk_size = config.mamba_cache_chunk_size;
    return out;
}

}  // namespace

Scheduler::Scheduler(SchedulerConfig config)
    : original_max_batch_size_{config.max_batch_size},
      config_{std::move(config)},
      device_allocator_{config_.page_size, config_.device_allocator.total_pages,
                        config_.enable_xpool_dynamic_capacity},
      host_allocator_{config_.page_size, config_.host_allocator.total_pages},
      mamba_allocator_{},
      kv_prefix_cache_{&device_allocator_, &host_allocator_,
                       config_.enable_l3_storage, config_.disable_prefix_cache, MakeEvictionConfig(config_)},
      req_pool_allocator_{config_.max_batch_size} {
    if (auto* env = std::getenv("SPDLOG_LEVEL")) {
        std::string level_str{env};
        spdlog::level::level_enum level = spdlog::level::from_str(level_str);
        spdlog::set_level(level);
    }

    if (config_.enable_kv_cache_events) {
        kv_prefix_cache_.SetKvEventSink([this](KvCacheEvent event) { kv_events_.push_back(std::move(event)); });
    }
    const bool has_mamba_pool = config_.enable_mamba && config_.mamba_pool_total_chunks > 0;
    if (has_mamba_pool) {
        mamba_allocator_.emplace(config_.mamba_pool_total_chunks,
                                  config_.enable_xpool_dynamic_capacity);
    }
    const bool has_mamba_l2_pool = has_mamba_pool && config_.enable_mamba_l2 && config_.mamba_l2_host_slots > 0;
    if (has_mamba_l2_pool) {
        mamba_host_allocator_.emplace(config_.mamba_l2_host_slots);
    }

    // Construct HybridPrefixCache when any adjunct/paged-cache feature is configured.
    // Role::kD skips Mamba but still participates in paged-cache transport.
    const bool has_mamba_adjunct = has_mamba_pool && config_.role != Role::kD;
    const bool has_prefix_cache_adjunct = config_.prefix_cache_adjunct.has_value();
    const bool has_paged_cache_groups = !config_.paged_cache_groups.empty();
    if (has_mamba_adjunct || has_prefix_cache_adjunct || has_paged_cache_groups) {
        MambaChunkAllocator* mamba_ptr = has_mamba_adjunct ? &*mamba_allocator_ : nullptr;
        MambaHostAllocator* mamba_host_ptr = has_mamba_l2_pool ? &*mamba_host_allocator_ : nullptr;
        hybrid_prefix_cache_.emplace(kv_prefix_cache_, mamba_ptr, config_.mamba_cache_chunk_size, mamba_host_ptr,
                                     MakeEvictionConfig(config_));
        kv_prefix_cache_.GetDeviceManager().SetEvictionCallback(
            [this](TreeNode* node) { hybrid_prefix_cache_->OnKVEvict(node); });
        kv_prefix_cache_.GetHostManager().SetEvictionCallback(
            [this](TreeNode* node) { hybrid_prefix_cache_->OnKVHostEvict(node); });
        // Prune frees TreeNodes (including empty ancestors) outside the per-tier
        // eviction callbacks; un-register them from the adjunct sets before the
        // node is destroyed so mamba_leaves_ / paged-cache membership never
        // dangles.
        kv_prefix_cache_.GetRadixTree().SetNodeDestroyCallback(
            [this](TreeNode* node) { hybrid_prefix_cache_->OnNodeDestroyed(node); });

        for (const auto& cfg : config_.paged_cache_groups) {
            PagedCacheGroupConfig copy = cfg;
            copy.Validate();
            hybrid_prefix_cache_->RegisterPagedCacheGroup(std::make_unique<PagedCacheGroupAllocator>(std::move(copy)));
        }

        if (has_prefix_cache_adjunct) {
            const auto& spec = *config_.prefix_cache_adjunct;
            if (spec.required_groups.empty()) {
                throw std::invalid_argument("Scheduler: prefix_cache_adjunct.required_groups must be non-empty");
            }
            // HybridPrefixCache derives history alignment from the registered
            // group configs; we still build the sliding-window map here.
            std::unordered_map<std::string, std::int32_t> sliding_window_per_group;
            for (const auto& gid : spec.required_groups) {
                const PagedCacheGroupConfig* cfg = nullptr;
                for (const auto& g : config_.paged_cache_groups) {
                    if (g.group_id == gid) {
                        cfg = &g;
                        break;
                    }
                }
                if (cfg == nullptr) {
                    throw std::invalid_argument("Scheduler: prefix_cache_adjunct required group_id '" + gid +
                                                "' not found in paged_cache_groups");
                }
                if (cfg->retention == PagedCacheGroupConfig::Retention::SlidingWindow) {
                    if (!cfg->sliding_window_tokens.has_value() || *cfg->sliding_window_tokens <= 0) {
                        throw std::invalid_argument("Scheduler: prefix_cache_adjunct sliding group '" + gid +
                                                    "' must declare positive sliding_window_tokens");
                    }
                    sliding_window_per_group.emplace(gid, *cfg->sliding_window_tokens);
                }
            }
            hybrid_prefix_cache_->EnablePagedCacheAdjunct(spec.required_groups, std::move(sliding_window_per_group));
        }
    }
    if (config_.enable_budgeter || config_.enable_admitter) {
        budget_agent_.emplace(config_);
    }

    // Dynamic-capacity mode: KV PageAllocator starts with zero mapped pages.
    // Grow to the profiled initial capacity so the engine serves requests right
    // away. When xpool_initial_kv_pages == 0 fall back to filling the full VA
    // window (total_pages - 1), which reproduces the static-mode baseline.
    //
    // The total_pages VA window should be larger than xpool_initial_kv_pages to
    // leave headroom for mamba→KV transfers. Setting
    //   total_pages = initial_kv + max_transfer_pages
    // (done on the Python side) is the recommended configuration.
    if (config_.enable_xpool_dynamic_capacity) {
        const std::int32_t max_pages = device_allocator_.TotalPages() - 1;
        const std::int32_t initial_kv =
            (config_.xpool_initial_kv_pages > 0 && config_.xpool_initial_kv_pages <= max_pages)
                ? config_.xpool_initial_kv_pages
                : max_pages;
        if (initial_kv > 0) {
            device_allocator_.Grow(initial_kv);
        }

        // Mamba allocator: also needs an initial Grow to its profiled baseline.
        if (mamba_allocator_) {
            const std::int32_t max_slots = config_.mamba_pool_total_chunks;
            const std::int32_t initial_mamba =
                (config_.xpool_initial_mamba_slots > 0 &&
                 config_.xpool_initial_mamba_slots <= max_slots)
                    ? config_.xpool_initial_mamba_slots
                    : max_slots;
            if (initial_mamba > 0) {
                mamba_allocator_->Grow(initial_mamba);
            }
        }
    }
}

std::vector<KvCacheEvent> Scheduler::DrainKvEvents() {
    std::vector<KvCacheEvent> events;
    events.swap(kv_events_);
    return events;
}

std::vector<std::string> Scheduler::CalcRollingHash(const std::vector<std::int32_t>& input_tokens, bool apply_match) {
    const std::int32_t page_size = config_.page_size;
    const std::size_t num_pages = input_tokens.size() / page_size;
    std::vector<std::span<const std::int32_t>> token_pages;
    token_pages.reserve(num_pages);
    for (std::size_t i = 0; i < num_pages; ++i) {
        token_pages.emplace_back(input_tokens.data() + i * page_size, page_size);
    }
    if (!apply_match) {
        return ComputePagedHashes(token_pages, "");
    }
    MatchResult result = kv_prefix_cache_.Match(token_pages);
    const std::int32_t host_matched = result.host.DepthInPage();
    if (host_matched >= static_cast<std::int32_t>(num_pages)) {
        return {};
    }
    const auto& hashes = result.host.last_node->PageHashes();
    std::string prior = hashes.empty() ? std::string{} : hashes.back();

    return ComputePagedHashes(
        std::vector<std::span<const std::int32_t>>(token_pages.begin() + host_matched, token_pages.end()), prior);
}

void Scheduler::SubmitRequests(const std::vector<RequestSpec>& request_specs) {
    for (const auto& spec : request_specs) {
        auto req = std::make_unique<Request>(spec, config_.page_size, config_.role);
        requests_.emplace(spec.request_id, std::move(req));
        if (budget_agent_) {
            budget_agent_->OnRequestArrival(static_cast<std::int32_t>(spec.tokens.size()), MakePoolSnapshot());
        }
    }
}

std::size_t Scheduler::WaitingSize() const {
    std::size_t count = 0;
    for (const auto& [id, req] : requests_) {
        if (req->Is<fsm::Submitted>()) {
            count++;
        }
    }
    return count;
}

std::size_t Scheduler::DecodingSize() const {
    std::size_t count = 0;
    for (const auto& [id, req] : requests_) {
        if (req->Is<fsm::Decoding>()) {
            count++;
        }
    }
    return count;
}

std::size_t Scheduler::PrefillSize() const {
    std::size_t count = 0;
    for (const auto& [id, req] : requests_) {
        if (req->Is<fsm::Prefilling>() || req->Is<fsm::PrefillDone>()) {
            count++;
        }
    }
    return count;
}

std::size_t Scheduler::RetractedSize() const {
    std::size_t count = 0;
    for (const auto& [id, req] : requests_) {
        if (req->Is<fsm::Retracting>() || req->Is<fsm::Retracted>()) {
            count++;
        }
    }
    return count;
}

std::size_t Scheduler::AvailableKvPages() const {
    return device_allocator_.AvailablePages();
}

std::size_t Scheduler::ActiveKvPages() const {
    std::unordered_set<std::int32_t> active_pages;
    for (const auto& [_, req] : requests_) {
        if (req->Is<fsm::Prefilling>() || req->Is<fsm::PrefillDone>() || req->Is<fsm::Decoding>()) {
            for (std::int32_t page : req->GetOccupiedPages()) {
                active_pages.insert(page);
            }
        }
    }
    return active_pages.size();
}

std::vector<std::string> Scheduler::PagedCacheGroupIds() const {
    if (!hybrid_prefix_cache_) return {};
    return hybrid_prefix_cache_->PagedCacheGroupIds();
}

std::int32_t Scheduler::PagedCacheGroupTotalPages(const std::string& group_id) const {
    if (!hybrid_prefix_cache_) {
        throw std::out_of_range("Scheduler::PagedCacheGroupTotalPages: group_id not configured");
    }
    return hybrid_prefix_cache_->PagedCacheGroupTotalPages(group_id);
}

std::int32_t Scheduler::PagedCacheGroupAvailablePages(const std::string& group_id) const {
    if (!hybrid_prefix_cache_) {
        throw std::out_of_range("Scheduler::PagedCacheGroupAvailablePages: group_id not configured");
    }
    return hybrid_prefix_cache_->PagedCacheGroupAvailablePages(group_id);
}

std::int64_t Scheduler::PagedCacheGroupFailedAllocCount(const std::string& group_id) const {
    if (!hybrid_prefix_cache_) {
        throw std::out_of_range("Scheduler::PagedCacheGroupFailedAllocCount: group_id not configured");
    }
    return hybrid_prefix_cache_->PagedCacheGroupFailedAllocCount(group_id);
}

std::vector<std::int32_t> Scheduler::GetRequestPagedCachePageIds(const std::string& request_id,
                                                                 const std::string& group_id) const {
    if (!hybrid_prefix_cache_) {
        throw std::out_of_range("Scheduler::GetRequestPagedCachePageIds: group_id not configured");
    }
    return hybrid_prefix_cache_->GetRequestPagedCachePageIds(request_id, group_id);
}

std::int32_t Scheduler::GetRequestPagedCacheBaseLogicalPage(const std::string& request_id,
                                                            const std::string& group_id) const {
    if (!hybrid_prefix_cache_) {
        throw std::out_of_range("Scheduler::GetRequestPagedCacheBaseLogicalPage: group_id not configured");
    }
    return hybrid_prefix_cache_->GetRequestPagedCacheBaseLogicalPage(request_id, group_id);
}

std::int32_t Scheduler::GetRequestTokenSize(const std::string& id) const {
    auto it = requests_.find(id);
    if (it == requests_.end()) {
        return -1;
    }
    return it->second->TokenSize();
}

std::vector<WriteBackOperation> Scheduler::newWriteBackOperation(
    std::unordered_map<std::string, std::unique_ptr<Request>>& requests) {
    std::vector<WriteBackOperation> ops;
    if (config_.disable_l2_cache) {
        return ops;
    }
    for (auto& [id, req] : requests) {
        if (!req->Is<fsm::Draining>()) continue;
        const auto& pages_to_transfer = req->GetPagesToTransfer<fsm::Draining>();

        if (!pages_to_transfer.empty()) {
            cache_op_id op_id = kv_prefix_cache_.AllocateCacheOpId();
            CacheOpSpec spec;
            spec.request_id = id;
            cache_op_tracker_[op_id] = std::move(spec);
            ops.push_back(WriteBackOperation{
                op_id, std::vector<TransferPair>(pages_to_transfer.begin(), pages_to_transfer.end())});
            req->Apply(fsm::CommitDrainingEvent{});
        } else {
            req->Apply(fsm::AbortEvent{});
        }
    }
    return ops;
}

ExecutionPlan Scheduler::NextExecutionPlan() {
    ExecutionPlan plan;

    std::vector<WriteBackOperation> write_back_ops;
    write_back_ops = std::move(newWriteBackOperation(requests_));

    if (hybrid_prefix_cache_) {
        for (const auto& [id, req] : requests_) {
            if (req->Is<fsm::Finished>()) {
                hybrid_prefix_cache_->ReleaseRequest(id);
            }
        }
    }
    std::erase_if(requests_, [](const auto& req) { return req.second->template Is<fsm::Finished>(); });

    std::vector<Request*> candidates;
    for (auto& [id, req] : requests_) {
        if (!req->Is<fsm::Draining>() && !req->Is<fsm::Prefetching>() && !req->Is<fsm::Retracting>() &&
            !req->Is<fsm::WritingBack>()) {
            candidates.push_back(req.get());
        }
    }

    auto [fwd_ops, cache_ops] = newForwardOperation(candidates);
    plan.With(FlatForwardOperation{std::move(fwd_ops)});

    // Merge retract write-backs (if any) into the Draining write-back list, then emit once.
    if (auto* wb = std::get_if<std::vector<WriteBackOperation>>(&cache_ops)) {
        write_back_ops.insert(write_back_ops.end(), std::make_move_iterator(wb->begin()),
                              std::make_move_iterator(wb->end()));
    }
    if (!write_back_ops.empty()) {
        plan.With(CacheOperation{FlatWriteBackOperation{write_back_ops}});
    }
    if (auto* lb = std::get_if<std::vector<LoadBackOperation>>(&cache_ops)) {
        if (!lb->empty()) {
            plan.With(CacheOperation{FlatLoadBackOperation{*lb}});
        }
    }
    if (std::getenv("DEBUG_MEM")) {
        check_device_mem();
    }
    return plan;
}

void Scheduler::check_device_mem() {
    bool ok = true;
    const std::int32_t total_device = device_allocator_.TotalPages() - 1;
    std::unordered_map<std::string, std::vector<std::int32_t>> req_pages_map;
    // page_id → (owner_req_id, state_name) for duplicate tail-page reporting
    std::unordered_map<std::int32_t, std::pair<std::string, std::string>> page_owner;

    for (auto& [id, req] : requests_) {
        std::string state = req->StateName();
        std::vector<std::int32_t> pages = req->GetLocalAllocatorPages();
        if (pages.empty()) continue;
        req_pages_map[id] = pages;

        for (std::int32_t p : pages) {
            auto [it, inserted] = page_owner.emplace(p, std::make_pair(id, state));
            if (!inserted) {
                spdlog::error("[check_mem] DEVICE TAIL PAGE OVERLAP: page={}  req1={}({})  req2={}({})", p,
                              it->second.first, it->second.second, id, state);
                ok = false;
            }
        }
    }

    // ── 2. Collect pages in radix tree ───────────────────────────────────────
    auto tree_device_pages = kv_prefix_cache_.CollectAllPages<ResourceType::Device>();

    // 2a. Check for duplicate page_ids inside the tree itself
    for (auto& [page, cnt] : tree_device_pages) {
        if (cnt > 1) {
            spdlog::error("[check_mem] DEVICE TREE DUPLICATE: page={} appears {} times in radix tree", page, cnt);
            ok = false;
        }
    }

    std::int32_t tree_device_total = static_cast<std::int32_t>(tree_device_pages.size());

    std::int32_t req_device_total = 0;
    for (auto& [id, pages] : req_pages_map) req_device_total += static_cast<std::int32_t>(pages.size());

    std::int32_t free_device = device_allocator_.AvailablePages();

    if (tree_device_total + req_device_total + free_device != total_device) {
        spdlog::error("[check_mem] DEVICE PAGE ACCOUNTING MISMATCH: tree={} req={} free={} sum={} total={}",
                      tree_device_total, req_device_total, free_device,
                      tree_device_total + req_device_total + free_device, total_device);
        ok = false;
    }

    // ── 4. Per-request: page ids must be in [1, total] ────────────────────
    // PageAllocator starts from page id 1 (0 is reserved as invalid/null).
    for (auto& [id, pages] : req_pages_map) {
        for (std::int32_t p : pages) {
            if (p <= 0 || p > total_device) {
                spdlog::error("[check_mem] INVALID DEVICE PAGE id={} for req={} (valid range [1,{}])", p, id,
                              total_device);
                ok = false;
            }
        }
    }
    for (auto& [p, cnt] : tree_device_pages) {
        if (p <= 0 || p > total_device) {
            spdlog::error("[check_mem] INVALID DEVICE PAGE id={} in radix tree (valid range [1,{}])", p, total_device);
            ok = false;
        }
    }

    // ── 5. Summary ────────────────────────────────────────────────────────────
    if (!ok) {
        throw std::runtime_error("Scheduler::CheckMem: device page accounting check failed");
    }
}

void Scheduler::Advance(const ExecutionEvent& event) {
    auto dispatch = [this](const auto& inner) { handleEvent(inner); };
    for (const auto& item : event.Events()) {
        std::visit([&](const auto& outer) { std::visit(dispatch, outer); }, item);
    }
}

PoolSnapshot Scheduler::MakePoolSnapshot() const {
    PoolSnapshot snapshot;
    snapshot.kv_free_pages = device_allocator_.AvailablePages();
    // Use MappedPages() as the denominator so that utilisation reflects only
    // physically-accessible capacity.  TotalPages() includes unmapped VMM
    // headroom which would create phantom pressure and confuse the budgeter.
    snapshot.kv_total_pages = device_allocator_.MappedPages();
    snapshot.kv_evictable_pages = kv_prefix_cache_.GetDeviceManager().EvictablePagesNum();
    snapshot.kv_headroom_pages = device_allocator_.HeadroomPages();
    if (mamba_allocator_) {
        snapshot.mamba_free_slots = mamba_allocator_->AvailableSlots();
        // Same reasoning: use MappedSlots() to avoid phantom mamba utilisation.
        snapshot.mamba_total_slots = mamba_allocator_->MappedSlots();
        snapshot.mamba_headroom_slots = mamba_allocator_->HeadroomSlots();
        // Evictable: prefix-cached mamba states that are not pinned by any
        // active request.  Counted here so the budgeter can subtract them from
        // mamba pressure — cached slots should not prevent mamba_to_kv fires.
        if (hybrid_prefix_cache_) {
            snapshot.mamba_evictable_slots = hybrid_prefix_cache_->MambaEvictableSlots();
        }
    }
    snapshot.queue_len = static_cast<std::int32_t>(WaitingSize());
    // PressureAdapter (S2.3): retracted_count is the number of requests in
    // fsm::Retracting/Retracted — a leading indicator that KV demand
    // recently exceeded supply.  BudgetAgent blends this into KV pressure
    // when config_.xpool_w_retract > 0 (default 0 = no behavioural change).
    snapshot.retracted_count = static_cast<std::int32_t>(RetractedSize());
    return snapshot;
}

void Scheduler::BudgetTick() {
    if (!budget_agent_) {
        return;
    }
    budget_agent_->Tick(MakePoolSnapshot());
    // S2.7 (HiMA Phase 3): dynamic admission cap.  After the budgeter
    // tick we may have committed kv_to_mamba / mamba_to_kv fires that
    // changed the Mamba pool size.  Clamp the effective max_batch_size
    // to the current mamba_total_slots so the scheduler stops admitting
    // more decoders than can fit in Mamba (preventing churn that would
    // otherwise immediately retract them).  The cap auto-restores on
    // mamba_to_kv fires that grow Mamba back.  Capped to
    // original_max_batch_size_ so we never overflow the ReqPoolAllocator.
    if (config_.enable_dynamic_admission_cap && mamba_allocator_) {
        const std::int32_t mamba_total =
            static_cast<std::int32_t>(mamba_allocator_->MappedSlots());
        SetMaxBatchSize(mamba_total);
    }
}

std::optional<XPoolFirePlan> Scheduler::PendingXPoolFire() const {
    if (!budget_agent_) {
        return std::nullopt;
    }
    return budget_agent_->PendingFire();
}

void Scheduler::PrepareKvToMambaFire(std::int32_t n_kv_pages) {
    if (!config_.enable_xpool_dynamic_capacity || n_kv_pages <= 0) {
        return;
    }
    // Cap the tail KV pages before physical unmap so no new allocations land
    // on pages that are about to be transferred to the mamba pool.
    device_allocator_.Shrink(n_kv_pages);
    kv_pre_shrunk_pages_ += n_kv_pages;
}

bool Scheduler::HasCappedKvInflight() const {
    return device_allocator_.CappedInflightPages() > 0;
}

void Scheduler::PrepareMambaToKvFire(std::int32_t n_mamba_chunks) {
    if (!config_.enable_xpool_dynamic_capacity || n_mamba_chunks <= 0 ||
        !mamba_allocator_) {
        return;
    }
    // The Python side passes n_mamba_chunks in physical VMM-chunk units
    // (e.g. 20 × 2 MB chunks).  The C++ mamba allocator tracks logical
    // *sequence* slots whose byte cost is config_.mamba_bytes_per_slot.
    // Convert: n_seq = n_vmm_chunks * CHUNK_SIZE / mamba_bytes_per_slot.
    // When one fire moves less than one full sequence slot (n_seq == 0) no
    // C++ cap/drain step is needed — the Python arena shrinks physically
    // without displacing any sequence-level allocation.
    std::int32_t n_seq_slots = n_mamba_chunks;  // fallback: treat 1 chunk == 1 slot
    if (config_.mamba_bytes_per_slot > 0 && config_.kv_bytes_per_page > 0) {
        // Re-derive the sequence-slot equivalent using the same formula as
        // the budgeter: n_seq = n_kv * kv_bytes / mamba_bytes.
        // n_mamba_chunks ≈ n_kv * kv_bytes / CHUNK_SIZE, so
        //   n_seq ≈ n_mamba_chunks * CHUNK_SIZE / mamba_bytes_per_slot.
        // We approximate CHUNK_SIZE from the ratio: if n_kv * kv_bpp produces
        // n_mamba_chunks VMM chunks, then CHUNK_SIZE ≈ n_kv * kv_bpp / n_mamba_chunks.
        // Simpler: use budgeter_pages_per_fire * kv_bpp / mamba_bps directly.
        n_seq_slots = static_cast<std::int32_t>(
            static_cast<std::int64_t>(config_.budgeter_pages_per_fire) *
            static_cast<std::int64_t>(config_.kv_bytes_per_page) /
            static_cast<std::int64_t>(config_.mamba_bytes_per_slot));
    }
    if (n_seq_slots <= 0) {
        // Each fire moves less than one full mamba sequence slot in bytes.
        // No C++ shrink/drain required — the physical unmap is safe because
        // no sequence allocations will be displaced.
        mamba_pre_shrunk_slots_ = 0;
        return;
    }
    // Cap the tail mamba sequence slots before physical unmap so no new
    // allocations land on slots that are about to be transferred to KV.
    mamba_allocator_->Shrink(n_seq_slots);
    mamba_pre_shrunk_slots_ += n_seq_slots;
}

bool Scheduler::HasCappedMambaInflight() const {
    if (!mamba_allocator_) {
        return false;
    }
    return mamba_allocator_->CappedInflightSlots() > 0;
}

void Scheduler::ApplyXPoolFire(const XPoolFirePlan& plan) {
    if (!config_.enable_xpool_dynamic_capacity) {
        return;
    }
    const std::int32_t n_kv_pages = static_cast<std::int32_t>(plan.page_ids.size());
    if (n_kv_pages <= 0) {
        return;
    }

    // Compute how many mamba slots correspond to n_kv_pages worth of KV memory.
    // Both byte-per-unit values are stored in the scheduler config and come from
    // the Python profiling step.
    std::int32_t n_mamba_slots = 0;
    if (config_.mamba_bytes_per_slot > 0 && config_.kv_bytes_per_page > 0) {
        n_mamba_slots = static_cast<std::int32_t>(
            static_cast<std::int64_t>(n_kv_pages) *
            static_cast<std::int64_t>(config_.kv_bytes_per_page) /
            static_cast<std::int64_t>(config_.mamba_bytes_per_slot));
    }

    if (plan.direction == "mamba_to_kv") {
        // Physical memory moved from mamba arena → KV arena (done by actuator).
        // C++: grow KV allocator; mamba Shrink may already have been done by
        // PrepareMambaToKvFire — only shrink the remainder to avoid double-shrink.
        device_allocator_.Grow(n_kv_pages);
        if (mamba_allocator_ && n_mamba_slots > 0) {
            const std::int32_t remaining = n_mamba_slots - mamba_pre_shrunk_slots_;
            if (remaining > 0) {
                mamba_allocator_->Shrink(remaining);
            }
        }
        mamba_pre_shrunk_slots_ = 0;
    } else if (plan.direction == "kv_to_mamba") {
        // KV Shrink may already have been done by PrepareKvToMambaFire.
        // Only shrink the remaining pages (if any) to avoid double-shrink.
        const std::int32_t remaining = n_kv_pages - kv_pre_shrunk_pages_;
        if (remaining > 0) {
            device_allocator_.Shrink(remaining);
        }
        kv_pre_shrunk_pages_ = 0;
        if (mamba_allocator_ && n_mamba_slots > 0) {
            mamba_allocator_->Grow(n_mamba_slots);
        }
    }

    // Clear the pending latch so the budgeter can emit new plans, and arm
    // the S2.2 reverse-direction cooldown so the opposite direction cannot
    // fire again in the immediate next tick (cancel path skips this).
    if (budget_agent_) {
        budget_agent_->OnFireCommitted(plan.direction);
        budget_agent_->ClearPendingFire();
    }
}

void Scheduler::CancelXPoolFire() {
    // Called when the Python actuator skips the physical VMM step (e.g. arena
    // headroom exhausted).  We must UNDO any C++ Shrink that was performed in
    // preparation, since no physical transfer happened.  Failing to restore
    // mapped_pages_ causes kv_util to appear near-zero (the denominator shrinks
    // with each cancelled fire) which prevents the budgeter from ever switching
    // to the mamba_to_kv direction.
    if (kv_pre_shrunk_pages_ > 0) {
        // Undo PrepareKvToMambaFire's Shrink: raise mapped_pages_ back and
        // return the tail pages to the free list so kv_util stays accurate.
        device_allocator_.Grow(kv_pre_shrunk_pages_);
        kv_pre_shrunk_pages_ = 0;
    }
    if (mamba_pre_shrunk_slots_ > 0 && mamba_allocator_) {
        // Undo PrepareMambaToKvFire's Shrink: restore mamba slot capacity.
        mamba_allocator_->Grow(mamba_pre_shrunk_slots_);
        mamba_pre_shrunk_slots_ = 0;
    }
    if (budget_agent_) {
        budget_agent_->ClearPendingFire();
    }
}

std::int32_t Scheduler::MappedKvPages() const {
    return device_allocator_.MappedPages();
}

std::int32_t Scheduler::AvailableMambaSlots() const {
    if (mamba_allocator_) {
        return mamba_allocator_->AvailableSlots();
    }
    return 0;
}

std::int32_t Scheduler::MappedMambaSlots() const {
    if (mamba_allocator_) {
        return mamba_allocator_->MappedSlots();
    }
    return 0;
}

void Scheduler::SetMaxBatchSize(std::int32_t new_cap) {
    // S2.7: clamp to [1, original_max_batch_size_].  Lower bound 1 keeps
    // the scheduler making forward progress even when Mamba briefly
    // drops to zero free slots (the cap controls *admission*, not the
    // ability to drain existing requests).  Upper bound prevents
    // overflowing the ReqPoolAllocator, which is sized at boot.
    const std::int32_t clamped =
        std::max<std::int32_t>(1, std::min<std::int32_t>(new_cap, original_max_batch_size_));
    config_.max_batch_size = clamped;
}

}  // namespace tokenspeed
