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

// Unit tests for MambaChunkAllocator dynamic-capacity mode introduced for
// HiMA inter-pool capacity transfer.

#include <gtest/gtest.h>

#include "resource/allocator/mamba_chunk_allocator.h"

namespace tokenspeed::test {

TEST(MambaDynCapTest, StaticModeUnchanged) {
    MambaChunkAllocator alloc(8, /*dynamic=*/false);
    EXPECT_EQ(alloc.AvailableSlots(), 8);
    EXPECT_EQ(alloc.TotalSlots(), 8);
    EXPECT_EQ(alloc.MappedSlots(), 8);
    EXPECT_FALSE(alloc.DynamicCapacityEnabled());

    auto slot = alloc.Allocate();
    ASSERT_TRUE(slot.has_value());
    EXPECT_EQ(slot->Index(), 0);  // min-heap: smallest first
    EXPECT_EQ(alloc.AvailableSlots(), 7);
}

TEST(MambaDynCapTest, DynamicModeStartsEmpty) {
    MambaChunkAllocator alloc(16, /*dynamic=*/true);
    EXPECT_EQ(alloc.AvailableSlots(), 0);
    EXPECT_EQ(alloc.MappedSlots(), 0);
    EXPECT_TRUE(alloc.DynamicCapacityEnabled());
    EXPECT_FALSE(alloc.Allocate().has_value());
}

TEST(MambaDynCapTest, GrowMakesSlotsAllocatable) {
    MambaChunkAllocator alloc(16, /*dynamic=*/true);
    auto grown = alloc.Grow(4);
    ASSERT_EQ(grown.size(), 4u);
    EXPECT_EQ(alloc.AvailableSlots(), 4);
    EXPECT_EQ(alloc.MappedSlots(), 4);

    // Allocate returns min-first (slot 0).
    auto slot = alloc.Allocate();
    ASSERT_TRUE(slot.has_value());
    EXPECT_EQ(slot->Index(), 0);
}

TEST(MambaDynCapTest, ShrinkCapsTailSlots) {
    MambaChunkAllocator alloc(16, /*dynamic=*/true);
    alloc.Grow(8);
    ASSERT_EQ(alloc.AvailableSlots(), 8);

    ASSERT_TRUE(alloc.Shrink(3));
    EXPECT_EQ(alloc.MappedSlots(), 5);
    EXPECT_EQ(alloc.AvailableSlots(), 5);
}

TEST(MambaDynCapTest, GrowAfterShrinkRestoresCapacity) {
    MambaChunkAllocator alloc(16, /*dynamic=*/true);
    alloc.Grow(8);
    alloc.Shrink(4);
    EXPECT_EQ(alloc.AvailableSlots(), 4);

    // Reclaim 2 slots back.
    auto grown = alloc.Grow(2);
    EXPECT_EQ(grown.size(), 2u);
    EXPECT_EQ(alloc.AvailableSlots(), 6);
    EXPECT_EQ(alloc.MappedSlots(), 6);
}

TEST(MambaDynCapTest, GrowRefusesToExceedTotal) {
    MambaChunkAllocator alloc(8, /*dynamic=*/true);
    EXPECT_THROW(alloc.Grow(100), std::runtime_error);
}

TEST(MambaDynCapTest, ShrinkRefusesToUnderflow) {
    MambaChunkAllocator alloc(8, /*dynamic=*/true);
    alloc.Grow(4);
    // Shrinking more than mapped should fail gracefully.
    EXPECT_FALSE(alloc.Shrink(100));
    EXPECT_EQ(alloc.MappedSlots(), 4);
}

TEST(MambaDynCapTest, CapAndUncapSlotsRoundTrip) {
    MambaChunkAllocator alloc(16, /*dynamic=*/true);
    alloc.Grow(8);
    const std::int32_t before = alloc.AvailableSlots();

    alloc.CapSlots({1, 3, 5});
    EXPECT_EQ(alloc.AvailableSlots(), before - 3);

    alloc.UncapSlots({1, 3, 5});
    EXPECT_EQ(alloc.AvailableSlots(), before);
}

TEST(MambaDynCapTest, StaticModeRejectsCapOps) {
    MambaChunkAllocator alloc(8, /*dynamic=*/false);
    EXPECT_THROW(alloc.CapSlots({1}), std::runtime_error);
    EXPECT_THROW(alloc.UncapSlots({1}), std::runtime_error);
}

TEST(MambaDynCapTest, FreeReturnsSlotsToAllocator) {
    MambaChunkAllocator alloc(8, /*dynamic=*/true);
    alloc.Grow(4);

    auto s0 = alloc.Allocate();
    ASSERT_TRUE(s0.has_value());
    EXPECT_EQ(alloc.AvailableSlots(), 3);

    s0.reset();  // triggers Free via RAII
    EXPECT_EQ(alloc.AvailableSlots(), 4);
}

}  // namespace tokenspeed::test
