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
#include <functional>
#include <optional>
#include <queue>
#include <vector>

#include "resource/allocator/capped_free_list.h"
#include "resource/radix_tree/mamba_slot.h"

namespace tokenspeed {

// MambaChunkAllocator manages recurrent-state slot indices for HiMA.
//
// Static mode (enable_dynamic_capacity=false):
//   All num_slots indices are immediately allocatable.  Uses a min-heap so
//   Allocate() always returns the smallest available index.  This guarantees
//   TP-rank determinism even when Free() calls arrive out-of-order.
//
// Dynamic mode (enable_dynamic_capacity=true):
//   Starts with zero mapped slots.  Call Grow() to expand the allocatable set
//   and Shrink() / CapSlots() / UncapSlots() to manage capacity transfers
//   for HiMA inter-pool fires.  Uses CappedFreeList (sorted) which preserves
//   the same min-first allocation order required for TP determinism.
class MambaChunkAllocator {
public:
    explicit MambaChunkAllocator(std::int32_t num_slots,
                                  bool enable_dynamic_capacity = false);

    std::optional<MambaSlot> Allocate();
    void Free(std::int32_t index);

    std::int32_t AvailableSlots() const;
    std::int32_t TotalSlots() const { return total_slots_; }
    std::int32_t MappedSlots() const { return mapped_slots_; }
    bool DynamicCapacityEnabled() const { return enable_dynamic_capacity_; }

    // --- Dynamic-capacity interface (no-op / error in static mode) ---

    // Grow by num_slots: returns the newly-available slot indices.
    std::vector<std::int32_t> Grow(std::int32_t num_slots);

    // Shrink by num_slots: caps tail slots so they are no longer allocatable.
    bool Shrink(std::int32_t num_slots);

    // Mark / unmark individual slots as capped (lent to the peer pool).
    void CapSlots(const std::vector<std::int32_t>& slot_ids);
    void UncapSlots(const std::vector<std::int32_t>& slot_ids);

private:
    std::int32_t total_slots_;
    std::int32_t mapped_slots_{0};
    bool enable_dynamic_capacity_{false};

    // Static mode: min-heap for TP rank determinism (Allocate = smallest free)
    std::priority_queue<std::int32_t, std::vector<std::int32_t>,
                        std::greater<std::int32_t>>
        free_list_;

    // Dynamic mode: sorted CappedFreeList — also returns min-first.
    CappedFreeList capped_free_list_{};
};

}  // namespace tokenspeed
