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

TEST(CappedFreeListTest, CappedPagesNeverReenterFreeList) {
    CappedFreeList list;
    list.Reset(8, {1, 2, 3, 4, 5, 6, 7});
    list.MarkCapped(3);
    auto page = list.Allocate();
    ASSERT_TRUE(page.has_value());
    EXPECT_NE(*page, 3);
    EXPECT_THROW(list.Deallocate(3), std::runtime_error);
}

TEST(CappedFreeListTest, GrowMakesPagesAllocatable) {
    PageAllocator alloc(4, 16, /*dynamic=*/true);
    auto grown = alloc.Grow(2);
    ASSERT_EQ(grown.size(), 2u);
    auto pages = alloc.Allocate(1);
    EXPECT_FALSE(pages.Empty());
}

}  // namespace tokenspeed::test
