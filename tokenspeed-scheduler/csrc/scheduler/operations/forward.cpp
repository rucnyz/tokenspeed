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

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <spdlog/spdlog.h>

#include "fsm/cache_states.h"
#include "fsm/forward_events.h"
#include "fsm/forward_states.h"
#include "resource/allocator/owned_pages.h"
#include "resource/allocator/req_pool_allocator.h"
#include "resource/radix_tree/node_range.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"
#include "resource/radix_tree/tree_node.h"
#include "resource/types.h"
#include "scheduler/operations/cache.h"
#include "scheduler/operations/forward.h"
#include "scheduler/request.h"
#include "scheduler/request_spec.h"
#include "scheduler/scheduler.h"
#include "scheduler/types.h"
#include "utils.h"

namespace tokenspeed {

namespace {

constexpr std::int32_t kLocalMambaSlotsPerRequest = 2;

std::int32_t CountMambaDeviceLoadBackSlots(const std::vector<TreeNode*>& nodes) {
    std::int32_t slots = 0;
    for (TreeNode* node : nodes) {
        if (node != nullptr && node->HasMambaOnHost() && !node->HasMamba()) {
            ++slots;
        }
    }
    return slots;
}

void AddUniqueNode(std::vector<TreeNode*>& nodes, TreeNode* node) {
    if (node == nullptr) return;
    if (std::find(nodes.begin(), nodes.end(), node) == nodes.end()) {
        nodes.push_back(node);
    }
}

}  // namespace

std::optional<fsm::SchedulePrefillFirstChunkEvent> Scheduler::schedulePrefillFirstChunk(
    Request* request, std::int32_t remaining, std::int32_t decode_input_tokens, bool disable_l2_cache,
    std::map<std::string, std::int32_t>& simulated_free) {
    if (req_pool_allocator_.AvailableSlots() == 0) return {};
    MatchResult match_result = hybrid_prefix_cache_ ? hybrid_prefix_cache_->Match(request->GetFullPagedTokens(true))
                                                    : kv_prefix_cache_.Match(request->GetFullPagedTokens(true));
    std::int32_t loadback_tokens = 0;
    std::int32_t unscheduled = 0;
    std::vector<TreeNode*> loadback_diff;
    std::vector<TreeNode*> mamba_loadback_nodes;

    const std::int32_t device_matched = match_result.device.DepthInPage();
    const std::int32_t host_matched = match_result.host.DepthInPage();
    if (disable_l2_cache) {
        unscheduled = request->PrefillSize() - device_matched * config_.block_size;
    } else {
        loadback_diff = match_result.NodesWithout<ResourceType::Device>();
        if (host_matched > device_matched) {
            loadback_tokens = config_.block_size * (host_matched - device_matched);
        }
        unscheduled = request->PrefillSize() - std::max(device_matched, host_matched) * config_.block_size;
    }

    std::int32_t tokens_this_round = std::min(remaining, unscheduled);
    if (hybrid_prefix_cache_ && hybrid_prefix_cache_->HasMambaAdjunct() && match_result.mamba_branching_seqlen == -1) {
        const std::int32_t aligned = hybrid_prefix_cache_->AlignMambaCacheSeqlen(tokens_this_round);
        if (aligned > 0) {
            match_result.mamba_branching_seqlen = aligned;
        }
    }

    std::int32_t num_tokens = loadback_tokens + tokens_this_round + decode_input_tokens;
    std::int32_t device_pages_needed = (num_tokens + config_.block_size - 1) / config_.block_size;

    std::unique_ptr<DeviceNodeRef> temp_lock = std::make_unique<DeviceNodeRef>(match_result.device.last_node);

    // Evict unlocked prefix-cache nodes before allocating request-local pages.
    if (!(kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Device>(device_pages_needed))) {
        return {};
    }

    if (hybrid_prefix_cache_ && hybrid_prefix_cache_->HasMambaAdjunct() && match_result.mamba_host_src_index >= 0 &&
        match_result.mamba_cow_src_index < 0) {
        TreeNode* host_mamba_node = hybrid_prefix_cache_->FindLastMambaHostNode(match_result.host.last_node);
        if (host_mamba_node != nullptr && host_mamba_node->HasMambaOnHost() && !host_mamba_node->HasMamba()) {
            AddUniqueNode(mamba_loadback_nodes, host_mamba_node);
        }
    }
    const bool needs_mamba_loadback = !mamba_loadback_nodes.empty();
    const std::int32_t mamba_loadback_slots_needed =
        needs_mamba_loadback ? CountMambaDeviceLoadBackSlots(mamba_loadback_nodes) : 0;
    const std::int32_t mamba_slots_needed = 2 + mamba_loadback_slots_needed;
    if (hybrid_prefix_cache_ && hybrid_prefix_cache_->HasMambaAdjunct() &&
        !hybrid_prefix_cache_->EnsureMambaCapacityByEvict(mamba_slots_needed)) {
        return {};
    }

    const std::int32_t first_pos = request->PrefillSize() - unscheduled;
    const std::int32_t target = first_pos + tokens_this_round;
    if (hybrid_prefix_cache_ &&
        !hybrid_prefix_cache_->AdmitChunk(request->Id(), first_pos, target, simulated_free, match_result.paged_cache)) {
        return {};
    }
    if (needs_mamba_loadback) {
        hybrid_prefix_cache_->PrepareMambaDeviceLoadBack(mamba_loadback_nodes);
        TreeNode* mamba_node = hybrid_prefix_cache_->FindLastMambaNode(match_result.host.last_node);
        if (mamba_node != nullptr) {
            match_result.mamba_cow_src_index = mamba_node->MambaSlotIndex();
        }
    }
    if (mamba_allocator_ && mamba_allocator_->AvailableSlots() < kLocalMambaSlotsPerRequest) {
        return {};
    }

    return fsm::SchedulePrefillFirstChunkEvent{
        tokens_this_round,
        decode_input_tokens,
        &device_allocator_,
        &req_pool_allocator_,
        match_result,
        config_.role,
        &kv_prefix_cache_,
        disable_l2_cache,
        std::move(loadback_diff),
        hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr,
        mamba_allocator_ ? &*mamba_allocator_ : nullptr,
        std::move(mamba_loadback_nodes),
    };
}

std::optional<fsm::SchedulePrefillEvent> Scheduler::schedulePrefill(
    Request* request, std::int32_t remaining, std::int32_t reserve_num_tokens_in_next_schedule_event,
    std::map<std::string, std::int32_t>& simulated_free) {
    std::int32_t unscheduled = request->UnScheduledPrefillSize();
    std::int32_t tokens_this_round = std::min(remaining, unscheduled);

    std::int32_t pages_needed = (tokens_this_round + config_.block_size - 1) / config_.block_size;

    if (!kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Device>(pages_needed)) {
        return {};
    }

    if (hybrid_prefix_cache_ && hybrid_prefix_cache_->HasMambaAdjunct() &&
        !hybrid_prefix_cache_->EnsureMambaCapacityByEvict(1)) {
        return {};
    }

    const std::int32_t first_pos = request->PrefillSize() - unscheduled;
    const std::int32_t target = first_pos + tokens_this_round;
    if (hybrid_prefix_cache_ && !hybrid_prefix_cache_->AdmitChunk(request->Id(), first_pos, target, simulated_free)) {
        return {};
    }

    return fsm::SchedulePrefillEvent{tokens_this_round, reserve_num_tokens_in_next_schedule_event,
                                     hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr};
}

std::optional<fsm::ScheduleDecodeEvent> Scheduler::scheduleDecode(Request* request,
                                                                  std::map<std::string, std::int32_t>& simulated_free) {
    std::int32_t tail_available = request->TailPageAvailableTokens();
    std::int32_t extra_tokens = std::max(0, request->GetReserveNumTokensInNextScheduleEvent() - tail_available);
    std::int32_t pages_needed = (extra_tokens + config_.block_size - 1) / config_.block_size;

    if (!kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Device>(pages_needed)) {
        return {};
    }

    if (hybrid_prefix_cache_ && hybrid_prefix_cache_->HasMambaAdjunct() && mamba_allocator_ &&
        request->Is<fsm::PrefillDone>() && request->GetLocalMambaAllocator() != nullptr &&
        !hybrid_prefix_cache_->EnsureMambaCapacityByEvict(1)) {
        return {};
    }

    const std::int32_t first_pos = request->TokenSize();
    const std::int32_t target = first_pos + config_.decode_input_tokens;
    if (hybrid_prefix_cache_ && !hybrid_prefix_cache_->AdmitChunk(request->Id(), first_pos, target, simulated_free)) {
        return {};
    }

    return fsm::ScheduleDecodeEvent{config_.decode_input_tokens,
                                    hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr};
}

std::optional<fsm::ScheduleDecodeFromRetractedEvent> Scheduler::scheduleDecodeFromRetracted(
    Request* request, std::map<std::string, std::int32_t>& simulated_free) {
    if (req_pool_allocator_.AvailableSlots() == 0) return {};

    MatchResult match_result =
        hybrid_prefix_cache_
            ? hybrid_prefix_cache_->Match(request->GetFullPagedTokens(true), MatchIntent::StateRecovery)
            : kv_prefix_cache_.Match(request->GetFullPagedTokens(true), MatchIntent::StateRecovery);
    std::vector<TreeNode*> loadback_diff = match_result.NodesWithout<ResourceType::Device>();
    std::vector<TreeNode*> mamba_loadback_nodes;
    TreeNode* mamba_recovery_node = nullptr;
    bool needs_mamba_loadback = false;
    if (hybrid_prefix_cache_ && mamba_allocator_) {
        mamba_recovery_node = hybrid_prefix_cache_->FindLastMambaNode(match_result.host.last_node);
        if (mamba_recovery_node == nullptr) {
            mamba_recovery_node = hybrid_prefix_cache_->FindLastMambaHostNode(match_result.host.last_node);
            needs_mamba_loadback = mamba_recovery_node != nullptr;
            if (needs_mamba_loadback && !mamba_recovery_node->HasMamba()) {
                AddUniqueNode(mamba_loadback_nodes, mamba_recovery_node);
            }
        }
        if (mamba_recovery_node == nullptr) {
            spdlog::warn("[Scheduler] Retracted request {} lost tree-owned Mamba state, aborting request",
                         request->Id());
            request->Apply(fsm::AbortEvent{});
            return {};
        }
        if (!needs_mamba_loadback) {
            match_result.mamba_cow_src_index = mamba_recovery_node->MambaSlotIndex();
        }
    }

    const std::int32_t device_matched2 = match_result.device.DepthInPage();
    const std::int32_t host_matched2 = match_result.host.DepthInPage();
    // Pages needed: LoadBack nodes (host→device) + pages for decode step itself.
    std::int32_t num_tokens = 0;
    if (host_matched2 > device_matched2) {
        num_tokens += (config_.block_size * (host_matched2 - device_matched2)) + config_.decode_input_tokens;
    } else {
        num_tokens += config_.decode_input_tokens;
    }
    std::int32_t device_pages_needed = (num_tokens + config_.block_size - 1) / config_.block_size;

    std::unique_ptr<DeviceNodeRef> temp_lock = std::make_unique<DeviceNodeRef>(match_result.device.last_node);
    if (!kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Device>(device_pages_needed)) {
        return {};
    }
    if (hybrid_prefix_cache_ && mamba_allocator_) {
        // Recovery COWs the tree-owned Mamba state into fresh request-local
        // working/checkpoint slots. Protect the source node only for this
        // allocation; retracted Mamba states are otherwise normal evictable
        // tree-owned cache entries.
        const std::int32_t mamba_slots_needed = 2 + CountMambaDeviceLoadBackSlots(mamba_loadback_nodes);
        if (!hybrid_prefix_cache_->EnsureMambaCapacityByEvict(mamba_slots_needed, mamba_recovery_node)) {
            return {};
        }
    }

    const std::int32_t target = request->TokenSize();
    if (hybrid_prefix_cache_ && !hybrid_prefix_cache_->AdmitChunkFromRetracted(request->Id(), target, simulated_free,
                                                                               match_result.paged_cache)) {
        return {};
    }
    if (needs_mamba_loadback) {
        hybrid_prefix_cache_->PrepareMambaDeviceLoadBack(mamba_loadback_nodes);
        if (mamba_recovery_node->HasMamba()) {
            match_result.mamba_cow_src_index = mamba_recovery_node->MambaSlotIndex();
        }
    }
    if (mamba_allocator_ && mamba_allocator_->AvailableSlots() < kLocalMambaSlotsPerRequest) {
        return {};
    }

    return fsm::ScheduleDecodeFromRetractedEvent{
        config_.decode_input_tokens,
        &device_allocator_,
        &req_pool_allocator_,
        &kv_prefix_cache_,
        std::move(match_result),
        loadback_diff,
        mamba_allocator_ ? &*mamba_allocator_ : nullptr,
        std::move(mamba_loadback_nodes),
    };
}

std::optional<fsm::ScheduleRetractEvent> Scheduler::scheduleRetract(Request* request) {
    // In-flight gate (2026-07-18): never retract a request that is part of
    // the dispatched-but-uncommitted forward batch reported via
    // SetInflightRequests(). Under overlap scheduling that batch's kernels
    // are still running: applyRetract would free the victim's mamba working
    // slot (reallocated while the kernel still writes its recurrent state)
    // and WriteBackDone would later drop its device tree ref (pages reused
    // or, with XPool fires, physically cuMemUnmap-ed while the kernel still
    // reads its KV prefix). Observed as CUDA illegal-memory-access seconds
    // after retracting an actively-decoding victim (long_horizon sys reps,
    // 2026-07-18). This is the single funnel for every retract path (S2.1
    // OOM fallback, S2.5-followup-2 proactive drain retract, S2.6 migrate),
    // so throwing here covers them all; newRetractOperation() catches the
    // throw and skips this tick, and by the NEXT planning call the batch has
    // committed, so the victim is retried with no kernel in flight.
    if (IsRequestInflight(request->Id())) {
        throw std::runtime_error(
            "scheduleRetract: victim is in the in-flight forward batch, skipping this attempt");
    }

    auto full_paged_tokens = request->GetFullPagedTokens(true);
    std::vector<std::int32_t> prefix_pages = DevicePagesFromRoot(request->GetDeviceNode());
    std::int32_t total_available = static_cast<std::int32_t>(request->GetOccupiedPages().size());

    // Overlap scheduling: ExtendResult may grow the token container before the
    // next Acquire runs. Clamp to the pages we actually have.
    if (total_available < static_cast<std::int32_t>(full_paged_tokens.size())) {
        full_paged_tokens.resize(total_available);
    }

    std::int32_t alloc_count =
        static_cast<std::int32_t>(full_paged_tokens.size()) - static_cast<std::int32_t>(prefix_pages.size());

    // `alloc_count` is expected to equal exactly the size of the request's
    // local (uncommitted) KV allocator, since full_paged_tokens is
    // conceptually prefix_pages ++ local_allocator_pages. A mismatch here
    // means `prefix_pages` (walked from root via DevicePagesFromRoot) and
    // `full_paged_tokens`/`total_available` disagree about how much of the
    // request's history is already shared in the tree -- this is the
    // confirmed root cause of two related crashes under long-running stress:
    // OwnedPages::TakeFirst throwing "count out of range" (plain LRU
    // eviction, no XPool) and a CUDA "illegal memory access" surfacing much
    // later in an unrelated KV-cache-write kernel (LRU eviction + XPool
    // dynamic capacity, HiMA sys_lru ablation). It got dramatically easier to
    // repro once S2.5-followup-2's proactive retract started calling
    // scheduleRetract() on nearly every scheduler tick instead of only via
    // the rare S2.1 out-of-memory fallback or S2.6 migrate paths (see
    // runs/hima_retract_three_arm_20260711*).
    //
    // NOTE (2026-07-12): an earlier version of this reconcile trimmed
    // `prefix_pages` down to (full_paged_tokens.size() - local_available)
    // and re-Insert()-ed under that fabricated depth whenever prefix_pages
    // over-counted relative to token-derived full pages. That corrupts the
    // shared tree: the real tree already has a node at the true (larger)
    // depth, so inserting a second, request-local page at a shallower
    // position desyncs later DevicePagesFromRoot() walks for *other*
    // requests sharing that prefix from the request-local allocator's true
    // size, surfacing as an unrelated `OwnedPages::TakeFirst: count out of
    // range` crash a few ticks later (see
    // runs/hima_pressure_three_arm_20260712/sys/rep0/run.log). We must never
    // fabricate `prefix_pages`; when it over-counts, just skip this tick.
    //
    // When local_available > alloc_count (except_last under-counts local tail
    // pages during overlap), keep the token-aligned alloc_count and take only
    // that many pages this tick — the classic OOM-retract path. This is safe
    // because it never touches `prefix_pages`, which still reflects the real
    // tree depth. The alloc_count == 0 / local_available > 0 signature (a
    // Decoding request holding only its working page) is likewise safe here:
    // the in-flight gate at the top of this function already guarantees the
    // victim has no kernel in flight, so the working page / mamba slot can
    // be released without racing a writer. (A 2026-07-18 grace-window
    // variant deferred this case by 500 ms instead; it livelocked the S2.1
    // OOM fallback at kv_util=1.0 because boundary-stalled victims never
    // resolve the signature on their own. The in-flight gate supersedes it.)
    //
    // If prefix_pages over-counts (alloc_count < 0), throw so callers skip
    // this tick without mutating state.
    const std::int32_t local_available =
        static_cast<std::int32_t>(request->GetLocalAllocatorPages().size());
    if (alloc_count != local_available) {
        const bool tolerant = local_available > alloc_count && alloc_count >= 0;
        if (tolerant) {
            spdlog::warn(
                "scheduleRetract: alloc_count mismatch for request {} "
                "(full_paged_tokens={} prefix_pages={} total_available={} "
                "alloc_count={} local_available={}); taking token-aligned "
                "alloc_count this tick",
                request->Id(), full_paged_tokens.size(), prefix_pages.size(), total_available, alloc_count,
                local_available);
        } else {
            throw std::runtime_error(
                "scheduleRetract: alloc_count/local_available race, skipping this attempt");
        }
    }

    if (alloc_count > local_available) {
        throw std::runtime_error(
            "scheduleRetract: alloc_count exceeds local_available after reconcile, skipping this attempt");
    }
    // alloc_count == 0 is a legitimate state (e.g. disable_prefix_cache, or a
    // request whose entire KV footprint is already committed to the tree
    // with nothing left in its local allocator) -- it just means there are no
    // device KV pages to move this tick. The retract may still be needed to
    // free the request's Mamba slot, so fall through instead of throwing;
    // only a genuinely negative alloc_count (prefix_pages outgrew
    // full_paged_tokens, which should be impossible) is an error.
    if (alloc_count < 0) {
        throw std::runtime_error("scheduleRetract: negative alloc_count, skipping this attempt");
    }

    // Check host writeback capacity BEFORE mutating request-local/tree state.
    // scheduleRetract() previously called TakeFirstPages()+Insert<Device>()
    // (irreversibly moving `alloc_count` pages from the request's local
    // allocator into the shared tree, without updating the request's
    // device_node_ref_) and only afterwards checked
    // EnsureCapacityByEvict<Host>(); bailing out of that check via `return
    // {}` left the request internally inconsistent -- its device_node_ref_
    // still pointed at the pre-insert (shallower) node while its local
    // allocator had already lost those pages, with no owner left to
    // reconcile it. Later ticks (this request's own decode-step
    // InsertHybridCache(), or another request's scheduleRetract()) would
    // then compute a `new_page_count`/`alloc_count` against this corrupted
    // bookkeeping and crash with `OwnedPages::TakeFirst: count out of
    // range`. See runs/hima_pressure_three_arm_20260712/sys/rep0/run.log.
    //
    // Every other scheduleXxx() in this file (e.g. schedulePrefillFirstChunk
    // above) follows the same rule: Match() + all EnsureCapacityByEvict/
    // EnsureMambaCapacityByEvict/AdmitChunk checks happen first, and only
    // once every gate passes do we mutate allocators/the tree. Mirror that
    // here: Match() against the tree as it exists today (before inserting
    // the alloc pages), derive `host_pages_needed` from the *would-be*
    // post-insert device depth (`full_paged_tokens.size()`, which by
    // construction equals prefix_pages.size() + alloc_count) versus today's
    // host depth, and only mutate once EnsureCapacityByEvict<Host> succeeds.
    MatchResult pre_match = kv_prefix_cache_.Match(full_paged_tokens, MatchIntent::StateRecovery);
    std::unique_ptr<HostNodeRef> temp_lock = std::make_unique<HostNodeRef>(pre_match.host.last_node);
    const std::int32_t host_matched_before = pre_match.host.DepthInPage();
    std::int32_t host_pages_needed = 0;
    if (static_cast<std::int32_t>(full_paged_tokens.size()) > host_matched_before) {
        host_pages_needed = static_cast<std::int32_t>(full_paged_tokens.size()) - host_matched_before;
    }

    if (!kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Host>(host_pages_needed)) {
        return {};
    }

    // TOCTOU guard against overlap scheduling. Everything above this point is
    // read-only w.r.t. this request's device pages and the shared device tree
    // (Match()/EnsureCapacityByEvict<Host> only touch host state), but under
    // overlap scheduling the *asynchronous* forward/prefill dispatched on a
    // previous tick can still be growing this request's local KV allocator
    // (ExtendResult appends pages) and a sibling request's retract/finish can
    // reshape the shared device tree between the entry-point snapshot of
    // `prefix_pages`/`alloc_count`/`local_available` and the mutation below.
    // If we Insert<Device>() with a now-stale `prefix_pages` (wrong tree
    // depth) or TakeFirstPages() more pages than the local allocator still
    // holds, we either desync later DevicePagesFromRoot() walks -- which
    // surfaces as an async CUDA illegal-memory-access in a downstream
    // attention kernel -- or throw OwnedPages::TakeFirst "count out of range".
    // Re-derive the accounting immediately before mutating and bail if it
    // shifted; newRetractOperation() catches this and retries the victim next
    // tick with a backoff, so a skipped attempt is always safe. The common
    // (no-race) path is unchanged: the re-read equals the snapshot and we fall
    // straight through.
    {
        std::vector<std::int32_t> prefix_pages_now = DevicePagesFromRoot(request->GetDeviceNode());
        const std::int32_t local_available_now =
            static_cast<std::int32_t>(request->GetLocalAllocatorPages().size());
        if (prefix_pages_now != prefix_pages || local_available_now != local_available) {
            throw std::runtime_error(
                "scheduleRetract: device tree/local allocator shifted under overlap between "
                "snapshot and mutation, skipping this attempt");
        }
    }

    if (alloc_count > 0) {
        OwnedPages alloc_pages = request->TakeFirstPages(alloc_count);
        kv_prefix_cache_.Insert<ResourceType::Device>(full_paged_tokens, prefix_pages, std::move(alloc_pages));
    }

    MatchResult match_result = kv_prefix_cache_.Match(full_paged_tokens, MatchIntent::StateRecovery);
    // Insert<Device>() only touches the device resource; the host side (and
    // therefore the node `temp_lock` protects) is unchanged, so re-lock the
    // (possibly identical) node returned by the post-insert Match() and keep
    // it alive for the returned event.
    temp_lock = std::make_unique<HostNodeRef>(match_result.host.last_node);

    return fsm::ScheduleRetractEvent{&kv_prefix_cache_, &host_allocator_, match_result,
                                     hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr};
}

LoadBackOperation GenerateLoadBackOp(const std::vector<TreeNode*>& diff, const std::vector<TreeNode*>& mamba_nodes,
                                     cache_op_id op_id) {
    std::vector<TransferPair> transfers;

    for (TreeNode* node : diff) {
        const auto& host_pages = node->Host().Pages();
        const auto& device_pages = node->Device().Pages();
        for (std::size_t i = 0; i < host_pages.size(); ++i) {
            transfers.push_back(TransferPair{CacheKind::kKV, host_pages[i], device_pages[i]});
        }
    }
    for (TreeNode* node : mamba_nodes) {
        if (node != nullptr && node->HasMambaOnHost() && node->HasMamba()) {
            transfers.push_back(TransferPair{CacheKind::kMamba, node->MambaHostSlotIndex(), node->MambaSlotIndex()});
        }
    }
    return LoadBackOperation{op_id, std::move(transfers)};
}

std::optional<WriteBackOperation> Scheduler::applyEventAndGenerateOp(Request* request,
                                                                     fsm::ScheduleRetractEvent event) {
    // Event applier builds the (device_page, host_page) pairs.
    request->Apply(std::move(event));

    const auto& pages_to_transfer = request->GetPagesToTransfer<fsm::Retracting>();
    if (pages_to_transfer.empty()) {
        // No copy needed; advance Retracting to Retracted without an op_id.
        request->Apply(
            fsm::WriteBackDoneEvent{&kv_prefix_cache_, hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr});
        return std::nullopt;
    }
    // Register op_id so WriteBackDone can route back.
    cache_op_id op_id = kv_prefix_cache_.AllocateCacheOpId();
    CacheOpSpec spec;
    spec.request_id = request->Id();
    cache_op_tracker_[op_id] = std::move(spec);
    return WriteBackOperation{op_id, std::vector<TransferPair>(pages_to_transfer.begin(), pages_to_transfer.end()),
                              true};
}

std::optional<WriteBackOperation> Scheduler::newRetractOperation(Request* retract_request) {
    // scheduleRetract()'s page accounting (full_paged_tokens vs. local
    // allocator size) can race with concurrent overlap-scheduling mutations
    // to the radix tree (see the alloc_count-mismatch warning it logs), and
    // this has more than one caller: the S2.1 "device memory exhausted"
    // fallback below, the S2.6 migrate path, and the S2.5-followup-2
    // proactive-retract path in newXPoolCappedDrainRetractOperations(). Guard
    // centrally here so every caller degrades to "skip this victim, retry
    // next tick" instead of taking down the whole engine.
    try {
        if (auto event = scheduleRetract(retract_request)) {
            if (auto op = applyEventAndGenerateOp(retract_request, std::move(*event))) {
                return std::move(*op);
            }
        } else {
            spdlog::warn("[Scheduler] Retract failed for request {}: host capacity exhausted, aborting request",
                         retract_request->Id());
            retract_request->Apply(fsm::AbortEvent{});
        }
    } catch (const std::exception& e) {
        spdlog::warn("[Scheduler] newRetractOperation for request {} threw ({}); skipping this tick",
                     retract_request->Id(), e.what());
        RecordXPoolProactiveRetractBackoff(retract_request->Id());
    }
    return std::nullopt;
}

// Apply event: state transfer + resource allocation
template <typename Event>
    requires(std::same_as<Event, fsm::SchedulePrefillFirstChunkEvent> || std::same_as<Event, fsm::SchedulePrefillEvent>)
static PrefillOperation applyPrefillEvent(Request* request, Event event) {
    std::int32_t begin = static_cast<std::int32_t>(request->GetOccupiedPages().size());
    request->Apply(event);
    std::vector<std::int32_t> all_pages = request->GetOccupiedPages();
    std::int32_t sz = static_cast<std::int32_t>(all_pages.size()) - begin;

    auto info = request->GetPrefillInfo();
    auto op = PrefillOperation{{
        .request_id = request->Id(),
        .request_pool_index = request->GetReqPoolIndex(),
        .input_length = info.extend_len,
        .occupied_pages = std::move(all_pages),
        .begin = begin,
        .size = sz,
        .prefill_length = request->PrefillSize(),
    }};
    op.input_ids = std::vector<std::int32_t>(info.input_ids.begin(), info.input_ids.end());
    op.shifted_input_ids = std::move(info.shifted_input_ids);
    op.extend_prefix_len = info.already_scheduled_len;

    auto* mamba = request->GetLocalMambaAllocator();
    if (mamba != nullptr && mamba->HasWorking()) {
        op.mamba_working_idx = mamba->WorkingIndex();
        if (mamba->HasCheckpoint()) {
            op.mamba_checkpoint_dst_idx = mamba->CheckpointIndex();
        }
    }

    return op;
}

PrefillOperation Scheduler::applyEventAndGenerateOp(Request* request, fsm::SchedulePrefillFirstChunkEvent event) {
    auto match = event.GetMatchResult();
    auto op = applyPrefillEvent(request, std::move(event));
    // Mamba fields only when adjunct is active.
    if (hybrid_prefix_cache_ && hybrid_prefix_cache_->HasMambaAdjunct()) {
        op.mamba_cow_src_idx = match.mamba_cow_src_index;
        op.mamba_branching_seqlen = match.mamba_branching_seqlen;
    }
    // Order: attach, acquire, populate. Attach before acquire so prior-chunk
    // tail pages commit into snapshots before Acquire's ReleaseSkipped frees them.
    if (hybrid_prefix_cache_) {
        hybrid_prefix_cache_->CommitChunk(op.request_id, const_cast<TreeNode*>(request->GetDeviceNode()));
        hybrid_prefix_cache_->AcquireForRequest(op.request_id, op.extend_prefix_len,
                                                op.extend_prefix_len + op.input_length, match.paged_cache);
        hybrid_prefix_cache_->PopulateOp(op);
    }
    return op;
}

PrefillOperation Scheduler::applyEventAndGenerateOp(Request* request, fsm::SchedulePrefillEvent event) {
    auto op = applyPrefillEvent(request, std::move(event));
    // Order: attach, acquire, populate (see SchedulePrefillFirstChunkEvent).
    if (hybrid_prefix_cache_) {
        hybrid_prefix_cache_->CommitChunk(op.request_id, const_cast<TreeNode*>(request->GetDeviceNode()));
        hybrid_prefix_cache_->AcquireForRequest(op.request_id, op.extend_prefix_len,
                                                op.extend_prefix_len + op.input_length);
        hybrid_prefix_cache_->PopulateOp(op);
    }
    return op;
}

template <typename Event>
    requires(std::same_as<Event, fsm::ScheduleDecodeEvent> ||
             std::same_as<Event, fsm::ScheduleDecodeFromRetractedEvent>)
static DecodeOperation applyDecodeEvent(Request* request, Event event, std::int32_t decode_input_tokens) {
    std::int32_t begin = static_cast<std::int32_t>(request->GetOccupiedPages().size());
    request->Apply(std::move(event));
    std::vector<std::int32_t> all_pages = request->GetOccupiedPages();
    std::int32_t sz = static_cast<std::int32_t>(all_pages.size()) - begin;

    auto op = DecodeOperation{{
        .request_id = request->Id(),
        .request_pool_index = request->GetReqPoolIndex(),
        .input_length = decode_input_tokens,
        .occupied_pages = std::move(all_pages),
        .begin = begin,
        .size = sz,
        .prefill_length = request->PrefillSize(),
    }};

    auto* mamba = request->GetLocalMambaAllocator();
    if (mamba != nullptr && mamba->HasWorking()) {
        op.mamba_working_idx = mamba->WorkingIndex();
        if (mamba->HasCheckpoint()) {
            op.mamba_checkpoint_dst_idx = mamba->CheckpointIndex();
        }
    }

    return op;
}

DecodeOperation Scheduler::applyEventAndGenerateOp(Request* request, fsm::ScheduleDecodeEvent event) {
    const bool need_bootstrap_token = request->Is<fsm::PrefillDone>() && config_.role == Role::kD;
    std::int32_t bootstrap_token = need_bootstrap_token ? request->GetLastToken() : -1;
    const std::int32_t first_pos = request->TokenSize();
    const bool came_from_prefill_done = request->Is<fsm::PrefillDone>();

    auto op = applyDecodeEvent(request, std::move(event), config_.decode_input_tokens);
    if (need_bootstrap_token) {
        op.decode_input_id = bootstrap_token;
    }
    // Order: attach, acquire, populate.
    if (hybrid_prefix_cache_) {
        if (came_from_prefill_done) {
            hybrid_prefix_cache_->CommitChunk(op.request_id, const_cast<TreeNode*>(request->GetDeviceNode()));
        }
        hybrid_prefix_cache_->AcquireForRequest(op.request_id, first_pos, first_pos + op.input_length);
        hybrid_prefix_cache_->PopulateOp(op);
    }
    return op;
}

DecodeOperation Scheduler::applyEventAndGenerateOp(Request* request, fsm::ScheduleDecodeFromRetractedEvent event) {
    const std::int32_t mamba_cow_src_index = event.GetMatchResult().mamba_cow_src_index;
    auto paged_cache_hit = event.GetMatchResult().paged_cache;
    request->Apply(std::move(event));
    if (!request->Is<fsm::Decoding>()) {
        throw std::logic_error(
            "Scheduler::applyEventAndGenerateOp: expected state=Decoding after loadback recovery; got state=" +
            request->StateName());
    }
    std::vector<std::int32_t> all_pages = request->GetOccupiedPages();
    std::int32_t sz = static_cast<std::int32_t>(all_pages.size());
    DecodeOperation op{{
        .request_id = request->Id(),
        .request_pool_index = request->GetReqPoolIndex(),
        .input_length = config_.decode_input_tokens,
        .occupied_pages = std::move(all_pages),
        .begin = 0,
        .size = sz,
    }};
    op.decode_input_id = request->GetLastToken();
    op.hist_token_len = request->TokenSize() - 1;
    op.mamba_cow_src_idx = mamba_cow_src_index;

    auto* mamba = request->GetLocalMambaAllocator();
    if (mamba != nullptr && mamba->HasWorking()) {
        op.mamba_working_idx = mamba->WorkingIndex();
        if (mamba->HasCheckpoint()) {
            op.mamba_checkpoint_dst_idx = mamba->CheckpointIndex();
        }
    }

    if (hybrid_prefix_cache_) {
        hybrid_prefix_cache_->ReleaseRequest(op.request_id);
        hybrid_prefix_cache_->AcquireForRequest(op.request_id, 0, request->TokenSize(), paged_cache_hit);
        hybrid_prefix_cache_->PopulateOp(op);
    }
    return op;
}

std::tuple<std::vector<ForwardOperation>, std::variant<std::vector<LoadBackOperation>, std::vector<WriteBackOperation>>>
Scheduler::newForwardOperation(std::vector<Request*> candidates) {
    auto priority = [&](const Request* req) -> int {
        if (req->Is<fsm::Prefilling>()) return 1;
        if (req->Is<fsm::Submitted>()) return 2;
        if (req->Is<fsm::Decoding>() || req->Is<fsm::PrefillDone>()) {
            // Decode-first if mixed-batch is enabled; prefill-first otherwise.
            return config_.enable_mixed_prefill_decode ? 0 : 3;
        }
        if (req->Is<fsm::Retracted>()) return 4;
        return 9;
    };
    // TP-determinism: tie-break on Request::Id() so the relative order within a
    // priority class is identical across ranks. requests_ is an unordered_map
    // keyed by string id; libstdc++ randomizes string hashing per process, so
    // without the tiebreaker each rank visits candidates in a different order
    // and — when token_budget / page / mamba-slot constraints are tight — picks
    // a different subset to schedule. That made forward_op None on some ranks
    // and non-None on others, deadlocking the next NCCL collective.
    std::sort(candidates.begin(), candidates.end(), [&](const auto& a, const auto& b) {
        int pa = priority(a), pb = priority(b);
        return pa != pb ? pa < pb : a->Id() < b->Id();
    });

    std::vector<ForwardOperation> ops;
    std::int32_t token_budget = config_.max_scheduled_tokens;
    bool pushed_prefill = false;
    auto push_op = [&](auto op, bool uses_pool_slot = false) {
        if (config_.role != Role::kD) {
            token_budget -= op.input_length;
        }
        if constexpr (std::is_same_v<std::decay_t<decltype(op)>, PrefillOperation>) {
            pushed_prefill = true;
        }
        ops.push_back(std::move(op));
    };
    std::vector<LoadBackOperation> loadback_ops;
    auto simulated_free =
        hybrid_prefix_cache_ ? hybrid_prefix_cache_->InitialSimulatedFree() : std::map<std::string, std::int32_t>{};
    for (Request* request : candidates) {
        if (token_budget <= 0 || config_.max_batch_size == ops.size()) break;

        if (request->Is<fsm::Prefilling>() && config_.role != Role::kD) {
            std::int32_t reserver_num_tokens = config_.role == Role::kP ? 0 : config_.decode_input_tokens;
            if (auto ev = schedulePrefill(request, token_budget, reserver_num_tokens, simulated_free)) {
                push_op(applyEventAndGenerateOp(request, *ev));
            }
        } else if (request->Is<fsm::Submitted>() || request->Is<fsm::PrefetchDone>()) {
            // PrefetchDone: host cache populated; treat same as Submitted for forward scheduling.
            std::int32_t decode_input_tokens = config_.role == Role::kP ? 0 : config_.decode_input_tokens;

            if (auto ev = schedulePrefillFirstChunk(request, token_budget, decode_input_tokens,
                                                    config_.disable_l2_cache, simulated_free)) {
                std::vector<TreeNode*> loadback_diff = ev->GetLoadbackDiff();
                std::vector<TreeNode*> mamba_loadback_nodes = ev->GetMambaLoadbackNodes();
                push_op(applyEventAndGenerateOp(request, std::move(*ev)), true);
                // will be empty when disable_l2_cache
                if (!loadback_diff.empty() || !mamba_loadback_nodes.empty()) {
                    cache_op_id op_id = kv_prefix_cache_.AllocateCacheOpId();
                    loadback_ops.push_back(GenerateLoadBackOp(loadback_diff, mamba_loadback_nodes, op_id));
                }
            }
        } else if (request->Is<fsm::PrefillDone>() || (request->Is<fsm::Decoding>() && config_.role != Role::kP)) {
            // If mixed-batch is disabled, skip ALL decode if any prefill was scheduled this round.
            // If mixed-batch is enabled, the priority sort puts decodes first, so this
            // branch is reached before any prefill push.
            if (!config_.enable_mixed_prefill_decode && pushed_prefill) break;

            if (auto ev = scheduleDecode(request, simulated_free)) {
                push_op(applyEventAndGenerateOp(request, *ev));
            }
        } else if (request->Is<fsm::Retracted>() && config_.role != Role::kP) {
            if (!config_.enable_mixed_prefill_decode && pushed_prefill) break;

            if (auto ev = scheduleDecodeFromRetracted(request, simulated_free)) {
                std::vector<TreeNode*> loadback_diff = ev->GetLoadbackDiff();
                std::vector<TreeNode*> mamba_loadback_nodes = ev->GetMambaLoadbackNodes();
                push_op(applyEventAndGenerateOp(request, std::move(*ev)));
                if (!loadback_diff.empty() || !mamba_loadback_nodes.empty()) {
                    cache_op_id op_id = kv_prefix_cache_.AllocateCacheOpId();
                    loadback_ops.push_back(GenerateLoadBackOp(loadback_diff, mamba_loadback_nodes, op_id));
                }
            }
        }
    }

    // If all active decode requests failed, device memory is exhausted: retract the longest one.
    if (ops.empty() && !candidates.empty()) {
        std::vector<Request*> retract_candidates;
        for (Request* req : candidates) {
            // Skip requests in the dispatched-but-uncommitted forward batch
            // (see SetInflightRequests): scheduleRetract would refuse them
            // anyway, and skipping here lets the fallback pick the longest
            // *eligible* victim instead of burning the tick on a doomed one.
            if (IsRequestInflight(req->Id())) {
                continue;
            }
            if ((req->Is<fsm::Decoding>() || (req->Is<fsm::PrefillDone>() && config_.role != Role::kD)) &&
                config_.role != Role::kP) {
                retract_candidates.push_back(req);
            }
        }
        if (!retract_candidates.empty()) {
            Request* victim =
                *std::max_element(retract_candidates.begin(), retract_candidates.end(),
                                  [](const Request* a, const Request* b) { return a->TokenSize() < b->TokenSize(); });
            std::vector<WriteBackOperation> wb_ops;
            if (auto op = newRetractOperation(victim)) {
                wb_ops.push_back(std::move(*op));
            }
            return {std::vector<ForwardOperation>{}, std::move(wb_ops)};
        }
    }

    return {std::move(ops), std::move(loadback_ops)};
}

}  // namespace tokenspeed
