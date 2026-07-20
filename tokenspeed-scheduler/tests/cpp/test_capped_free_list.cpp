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

#include <gtest/gtest.h>

#include <unordered_set>
#include <utility>
#include <vector>

#include "resource/allocator/capped_free_list.h"
#include "resource/allocator/mamba_chunk_allocator.h"
#include "resource/allocator/owned_pages.h"
#include "resource/allocator/page_allocator.h"
#include "resource/radix_tree/mamba_slot.h"

namespace tokenspeed::test {

// Capped pages are drained (not re-added to the free list) when Deallocate is
// called.  This allows in-flight requests to "return" capped pages safely and
// lets InFlightCappedCount() reach zero once all holders have finished.
TEST(CappedFreeListTest, CappedPagesDrainedOnDeallocate) {
    CappedFreeList list;
    list.Reset(8, {1, 2, 3, 4, 5, 6, 7});
    list.MarkCapped(3);
    // Capped page must not be returned by Allocate.
    auto page = list.Allocate();
    ASSERT_TRUE(page.has_value());
    EXPECT_NE(*page, 3);

    // Simulate an in-flight request that was already holding page 3 calling
    // Deallocate after the page was capped.  Should NOT throw.
    EXPECT_NO_THROW(list.Deallocate(3));

    // After drain, InFlightCappedCount should be 0 (page 3 was in marks_; it
    // was free when capped so it already counted as drained via MarkCapped,
    // but calling Deallocate again is idempotent).
    EXPECT_EQ(list.InFlightCappedCount(), 0);

    // Page 3 must still not appear in the free list.
    bool got_three = false;
    for (int i = 0; i < 6; ++i) {
        auto p = list.Allocate();
        if (p.has_value() && *p == 3) got_three = true;
    }
    EXPECT_FALSE(got_three);
}

TEST(CappedFreeListTest, InFlightCappedCountDecreasesAsInflightPagesReturn) {
    CappedFreeList list;
    // 5 pages; none initially free (simulate all are in-flight).
    list.Reset(5, {});
    // Simulate: page 2 and 3 are in-flight (allocated externally), then capped.
    list.MarkCapped(2);
    list.MarkCapped(3);
    // Both were not in free_ids_ so they are truly in-flight (not drained yet).
    EXPECT_EQ(list.InFlightCappedCount(), 2);

    // One request finishes and returns page 2.
    EXPECT_NO_THROW(list.Deallocate(2));
    EXPECT_EQ(list.InFlightCappedCount(), 1);

    // Second request finishes and returns page 3.
    EXPECT_NO_THROW(list.Deallocate(3));
    EXPECT_EQ(list.InFlightCappedCount(), 0);
}

// The ordered-set free list must hand out the smallest available id first,
// regardless of the order pages are returned. This "smallest first" policy is
// what lets Shrink() cap the high tail without hitting in-flight pages so a
// kv_to_mamba fire can drain quickly (see the 2026-07-19 O(log n) rewrite that
// replaced the sorted std::vector). A regression to LIFO (pop_back) order would
// silently slow fire drains.
TEST(CappedFreeListTest, AllocateReturnsSmallestIdFirstAfterOutOfOrderDeallocate) {
    CappedFreeList list;
    list.Reset(8, {});
    // Return ids out of order; Allocate must still yield them ascending.
    list.Deallocate(5);
    list.Deallocate(1);
    list.Deallocate(7);
    list.Deallocate(3);
    std::vector<std::int32_t> got;
    for (;;) {
        auto p = list.Allocate();
        if (!p.has_value()) break;
        got.push_back(*p);
    }
    EXPECT_EQ(got, (std::vector<std::int32_t>{1, 3, 5, 7}));
}

TEST(CappedFreeListTest, GrowMakesPagesAllocatable) {
    PageAllocator alloc(4, 16, /*dynamic=*/true);
    auto grown = alloc.Grow(2);
    ASSERT_EQ(grown.size(), 2u);
    auto pages = alloc.Allocate(1);
    EXPECT_FALSE(pages.Empty());
}

// Regression: CancelXPoolFire undoes a PrepareKvToMambaFire Shrink via Grow.
// When the fire was cancelled because the drain TIMED OUT, pages in the
// re-grown tail are still owned by in-flight requests. Grow must not push
// those onto the free list: doing so hands the same KV page to a second
// request, and the two writers corrupt each other -- observed in production
// as a delayed CUDA illegal-memory-access ~28 s after a cancelled fire
// (cc_qwen_t6 @ max_total_tokens=640000, sys arm rep 2).
TEST(CappedFreeListTest, GrowAfterCancelledShrinkSkipsInflightPages) {
    PageAllocator alloc(4, 16, /*dynamic=*/true);
    alloc.Grow(10);  // pages 1..10 live

    // 8 pages go in-flight (held by requests). 2 stay free.
    OwnedPages held = alloc.Allocate(8);
    ASSERT_EQ(held.Size(), 8);

    // PrepareKvToMambaFire: tail-cap 4 pages (7..10). Given allocation order,
    // some of them are held by `held`, so the drain would not complete.
    ASSERT_TRUE(alloc.Shrink(4));
    const std::int32_t inflight_before = alloc.CappedInflightPages();
    ASSERT_GT(inflight_before, 0);

    // Drain timeout -> CancelXPoolFire -> Grow(4) to undo the Shrink.
    alloc.Grow(4);

    // Exhaust the allocator; nothing it hands out may overlap `held`.
    std::unordered_set<std::int32_t> held_ids(held.Ids().begin(), held.Ids().end());
    std::vector<OwnedPages> rest;
    for (;;) {
        OwnedPages p = alloc.Allocate(1);
        if (p.Empty()) break;
        EXPECT_EQ(held_ids.count(p.Ids().front()), 0u)
            << "page " << p.Ids().front() << " double-allocated while still in flight";
        rest.push_back(std::move(p));
    }
    // 10 live pages, 8 in flight -> at most 2 fresh ones.
    EXPECT_EQ(rest.size(), 2u);

    // Once the in-flight owner releases, its pages become allocatable again
    // (cap barrier was raised by Grow), so capacity is not leaked.
    rest.clear();
    { OwnedPages release = std::move(held); }
    OwnedPages after = alloc.Allocate(8);
    EXPECT_EQ(after.Size(), 8);
}

// Same guard on the mamba side: CancelXPoolFire also undoes
// PrepareMambaToKvFire via MambaChunkAllocator::Grow.
TEST(CappedFreeListTest, MambaGrowAfterCancelledShrinkSkipsInflightSlots) {
    MambaChunkAllocator alloc(/*num_slots=*/12, /*enable_dynamic_capacity=*/true);
    alloc.Grow(8);  // slots 0..7 live

    // MambaSlot is RAII: keeping the objects alive keeps the slots in flight.
    std::vector<MambaSlot> held;
    std::unordered_set<std::int32_t> held_ids;
    for (int i = 0; i < 6; ++i) {
        auto slot = alloc.Allocate();
        ASSERT_TRUE(slot.has_value());
        held_ids.insert(slot->Index());
        held.push_back(std::move(*slot));
    }

    ASSERT_TRUE(alloc.Shrink(4));  // tail-cap slots 4..7 (some held)
    ASSERT_GT(alloc.CappedInflightSlots(), 0);

    alloc.Grow(4);  // cancel path

    std::vector<MambaSlot> rest;
    for (;;) {
        auto slot = alloc.Allocate();
        if (!slot.has_value()) break;
        EXPECT_EQ(held_ids.count(slot->Index()), 0u)
            << "slot " << slot->Index() << " double-allocated while still in flight";
        rest.push_back(std::move(*slot));
    }
    EXPECT_EQ(rest.size(), 2u);  // 8 live - 6 held

    // Releasing the in-flight holders (RAII Free) returns their slots to the
    // pool: capacity is not leaked by the skip-in-flight guard.
    held.clear();
    std::int32_t reclaimed = 0;
    std::vector<MambaSlot> reclaimed_slots;
    for (;;) {
        auto slot = alloc.Allocate();
        if (!slot.has_value()) break;
        ++reclaimed;
        reclaimed_slots.push_back(std::move(*slot));
    }
    EXPECT_EQ(reclaimed, 6);
}

}  // namespace tokenspeed::test
