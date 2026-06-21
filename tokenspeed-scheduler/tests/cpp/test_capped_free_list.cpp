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

#include "resource/allocator/capped_free_list.h"
#include "resource/allocator/owned_pages.h"
#include "resource/allocator/page_allocator.h"

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

TEST(CappedFreeListTest, GrowMakesPagesAllocatable) {
    PageAllocator alloc(4, 16, /*dynamic=*/true);
    auto grown = alloc.Grow(2);
    ASSERT_EQ(grown.size(), 2u);
    auto pages = alloc.Allocate(1);
    EXPECT_FALSE(pages.Empty());
}

}  // namespace tokenspeed::test
