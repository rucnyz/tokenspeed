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

#include <cstdint>
#include <optional>
#include <set>
#include <unordered_set>
#include <vector>

namespace tokenspeed {

// Cap-barrier aware free list for HiMA-style cross-pool transfer. Capped page ids
// never appear in the allocatable free list; tail_lo tracks a contiguous capped
// suffix without materializing it.
class CappedFreeList {
public:
    static constexpr std::int32_t kNoTail = -1;

    void Reset(std::int32_t size, std::vector<std::int32_t> initial_free);

    std::optional<std::int32_t> Allocate();
    void Deallocate(std::int32_t page_id);

    void MarkCapped(std::int32_t page_id);
    void UnmarkCapped(std::int32_t page_id);
    void SetCap(std::int32_t tail_lo);

    bool IsCapped(std::int32_t page_id) const;
    // True when page_id is capped and has already been freed by its owner
    // (present in the drained set). Capped pages NOT drained are still owned
    // by an in-flight request.
    bool IsDrained(std::int32_t page_id) const { return capped_drained_.count(page_id) > 0; }
    std::int32_t Live() const;
    std::int32_t Available() const { return static_cast<std::int32_t>(free_ids_.size()); }
    // (kept O(log n): free_ids_ is an ordered set, so size() is O(1).)
    std::int32_t NumCapped() const { return n_capped_; }
    std::int32_t Size() const { return size_; }

    // Returns the number of capped pages that are still allocated (in-flight).
    // Zero means all capped pages have been freed and it is safe to unmap.
    std::int32_t InFlightCappedCount() const;

private:
    bool inTail(std::int32_t page_id) const;

    std::int32_t size_{0};
    std::int32_t n_capped_{0};
    std::int32_t tail_lo_{kNoTail};
    // Ordered set of allocatable (uncapped, free) ids. An ordered set keeps the
    // "smallest id first" allocation policy -- which lets Shrink() cap the high
    // tail without hitting in-flight pages, so fires drain quickly -- while
    // making Allocate/Deallocate/MarkCapped all O(log n). The previous sorted
    // std::vector paid O(n) per erase-from-front and O(n log n) per Deallocate
    // (push_back + full std::sort); with pools of tens of thousands of pages
    // that sort ran on the single-threaded scheduler hot path for every page
    // freed (each decode step, cache eviction, and request completion),
    // dominating throughput and scaling with pool size. This is why the
    // dynamic-capacity path regressed vs. the static O(1) free list even when
    // no cross-pool fire was ever issued (idle overhead).
    std::set<std::int32_t> free_ids_{};
    std::unordered_set<std::int32_t> marks_{};
    // Tracks capped pages that have been returned via Deallocate (drained).
    // InFlightCappedCount = n_capped_ - capped_drained_.size().
    std::unordered_set<std::int32_t> capped_drained_{};
};

}  // namespace tokenspeed
