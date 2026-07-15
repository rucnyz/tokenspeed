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

// Capacity-accounting tests for the HiMA XPool data-plane "brain" on the KV
// side (PageAllocator dynamic-capacity mode). These cover the baseline restore
// performed by the scheduler at startup and the Shrink/Grow transfer sequence
// the actuator will drive once tensor-level VA binding is in place.

#include <gtest/gtest.h>

#include "resource/allocator/owned_pages.h"
#include "resource/allocator/page_allocator.h"

namespace tokenspeed::test {

namespace {

// Mirror the scheduler's startup baseline restore for dynamic-capacity mode.
void RestoreFullBaseline(PageAllocator& alloc) {
    const std::int32_t total = alloc.TotalPages();
    if (total > 1) {
        alloc.Grow(total - 1);
    }
}

}  // namespace

TEST(XPoolCapacityTest, InitialGrowMatchesStaticBaseline) {
    // Dynamic mode starts empty; the scheduler restores the full profiled share.
    PageAllocator dyn(/*page_size=*/4, /*total_pages=*/16, /*dynamic=*/true);
    EXPECT_EQ(dyn.AvailablePages(), 0);

    RestoreFullBaseline(dyn);

    // Static mode exposes pages [1, total) — dynamic baseline must match.
    PageAllocator stat(/*page_size=*/4, /*total_pages=*/16, /*dynamic=*/false);
    EXPECT_EQ(dyn.AvailablePages(), stat.AvailablePages());
    EXPECT_EQ(dyn.AvailablePages(), 15);
    EXPECT_EQ(dyn.MappedPages(), 15);
}

TEST(XPoolCapacityTest, ShrinkLendsTailPagesToOtherPool) {
    PageAllocator alloc(/*page_size=*/4, /*total_pages=*/16, /*dynamic=*/true);
    RestoreFullBaseline(alloc);
    ASSERT_EQ(alloc.AvailablePages(), 15);

    // kv_to_mamba: lend 4 tail pages. They must leave the KV free list.
    ASSERT_TRUE(alloc.Shrink(4));
    EXPECT_EQ(alloc.MappedPages(), 11);
    EXPECT_EQ(alloc.AvailablePages(), 11);

    // mamba_to_kv: reclaim 2 pages back into the KV pool.
    auto grown = alloc.Grow(2);
    EXPECT_EQ(grown.size(), 2u);
    EXPECT_EQ(alloc.MappedPages(), 13);
    EXPECT_EQ(alloc.AvailablePages(), 13);
}

TEST(XPoolCapacityTest, ShrinkRefusesToUnderflow) {
    PageAllocator alloc(/*page_size=*/4, /*total_pages=*/8, /*dynamic=*/true);
    RestoreFullBaseline(alloc);
    // Cannot shrink below the one reserved (null) page.
    EXPECT_FALSE(alloc.Shrink(100));
    EXPECT_EQ(alloc.AvailablePages(), 7);
}

TEST(XPoolCapacityTest, GrowRefusesToExceedReservation) {
    PageAllocator alloc(/*page_size=*/4, /*total_pages=*/8, /*dynamic=*/true);
    alloc.Grow(3);
    EXPECT_EQ(alloc.MappedPages(), 3);
    // Reserved VA window is total-1 == 7 pages; growing past it must throw.
    EXPECT_THROW(alloc.Grow(100), std::runtime_error);
}

TEST(XPoolCapacityTest, CapAndUncapRoundTrip) {
    PageAllocator alloc(/*page_size=*/4, /*total_pages=*/16, /*dynamic=*/true);
    RestoreFullBaseline(alloc);
    const std::int32_t before = alloc.AvailablePages();

    // Cap individual pages (physical memory handed to the other pool); they
    // must drop out of the allocatable set, then return after uncapping.
    alloc.CapPages({2, 5, 7});
    EXPECT_EQ(alloc.AvailablePages(), before - 3);

    alloc.UncapPages({2, 5, 7});
    EXPECT_EQ(alloc.AvailablePages(), before);
}

TEST(XPoolCapacityTest, ShrinkMarksTailPagesCapped) {
    PageAllocator alloc(/*page_size=*/4, /*total_pages=*/16, /*dynamic=*/true);
    RestoreFullBaseline(alloc);
    ASSERT_TRUE(alloc.Shrink(4));
    EXPECT_TRUE(alloc.IsPageCapped(12));
    EXPECT_TRUE(alloc.IsPageCapped(15));
    EXPECT_FALSE(alloc.IsPageCapped(11));
}

TEST(XPoolCapacityTest, AllocatePartialFailureDoesNotLeakPages) {
    // Regression test: CappedFreeList::Allocate() pops (erases) a page id
    // from free_ids_ before PageAllocator::Allocate() knows the whole request
    // can be satisfied. If a later page in the same request turns out
    // unavailable (all remaining free ids capped), the pages already popped
    // this call must be rolled back into the free list — otherwise they leak
    // permanently and AvailablePages()/free capacity silently shrink every
    // time a request races the free list running dry.
    PageAllocator alloc(/*page_size=*/4, /*total_pages=*/16, /*dynamic=*/true);
    RestoreFullBaseline(alloc);
    ASSERT_EQ(alloc.AvailablePages(), 15);

    // Cap all but 3 pages so a request for more than 3 pages can only
    // partially succeed before CappedFreeList::Allocate() starts returning
    // nullopt (every remaining free id is capped).
    alloc.CapPages({4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    ASSERT_EQ(alloc.AvailablePages(), 3);

    // Requesting 5 pages must fail outright (only 3 uncapped pages exist),
    // and must not silently consume any of the 3 uncapped pages doing so.
    OwnedPages failed = alloc.Allocate(5);
    EXPECT_TRUE(failed.Empty());
    EXPECT_EQ(alloc.AvailablePages(), 3);

    // The 3 pages must still be genuinely allocatable afterwards.
    OwnedPages ok = alloc.Allocate(3);
    EXPECT_EQ(ok.Size(), 3);
    EXPECT_EQ(alloc.AvailablePages(), 0);
}

TEST(XPoolCapacityTest, StaticModeRejectsCapOperations) {
    PageAllocator alloc(/*page_size=*/4, /*total_pages=*/8, /*dynamic=*/false);
    EXPECT_THROW(alloc.CapPages({1}), std::runtime_error);
    EXPECT_THROW(alloc.UncapPages({1}), std::runtime_error);
}

}  // namespace tokenspeed::test
