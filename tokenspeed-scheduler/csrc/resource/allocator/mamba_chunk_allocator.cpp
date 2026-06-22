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

#include "resource/allocator/mamba_chunk_allocator.h"

#include <stdexcept>

namespace tokenspeed {

MambaSlot::MambaSlot(std::int32_t index, MambaChunkAllocator* allocator)
    : MambaSlot(index, [allocator](std::int32_t i) {
          if (allocator != nullptr) allocator->Free(i);
      }) {}

MambaChunkAllocator::MambaChunkAllocator(std::int32_t num_slots,
                                          bool enable_dynamic_capacity)
    : total_slots_{num_slots},
      mapped_slots_{enable_dynamic_capacity ? 0 : num_slots},
      enable_dynamic_capacity_{enable_dynamic_capacity} {
    if (enable_dynamic_capacity_) {
        // Dynamic mode: start empty; caller uses Grow() to expand.
        capped_free_list_.Reset(total_slots_, {});
    } else {
        for (std::int32_t i = 0; i < num_slots; ++i) {
            free_list_.push(i);
        }
    }
}

std::int32_t MambaChunkAllocator::AvailableSlots() const {
    if (enable_dynamic_capacity_) {
        return capped_free_list_.Available();
    }
    return static_cast<std::int32_t>(free_list_.size());
}

std::optional<MambaSlot> MambaChunkAllocator::Allocate() {
    if (enable_dynamic_capacity_) {
        auto slot = capped_free_list_.Allocate();
        if (!slot.has_value()) {
            return std::nullopt;
        }
        return MambaSlot{*slot, this};
    }
    if (free_list_.empty()) {
        return std::nullopt;
    }
    std::int32_t index = free_list_.top();
    free_list_.pop();
    return MambaSlot{index, this};
}

void MambaChunkAllocator::Free(std::int32_t index) {
    if (enable_dynamic_capacity_) {
        capped_free_list_.Deallocate(index);
        return;
    }
    free_list_.push(index);
}

std::vector<std::int32_t> MambaChunkAllocator::Grow(std::int32_t num_slots) {
    if (!enable_dynamic_capacity_ || num_slots <= 0) {
        return {};
    }
    if (mapped_slots_ + num_slots > total_slots_) {
        throw std::runtime_error("MambaChunkAllocator::Grow: exceeds reserved capacity");
    }
    const std::int32_t old_mapped = mapped_slots_;
    mapped_slots_ += num_slots;
    // Raise the cap barrier above the newly mapped range so slots that were
    // previously capped by a Shrink become allocatable again.
    capped_free_list_.SetCap(mapped_slots_);
    std::vector<std::int32_t> grown;
    grown.reserve(static_cast<std::size_t>(num_slots));
    for (std::int32_t id = old_mapped; id < mapped_slots_; ++id) {
        capped_free_list_.Deallocate(id);
        grown.push_back(id);
    }
    return grown;
}

bool MambaChunkAllocator::Shrink(std::int32_t num_slots) {
    if (!enable_dynamic_capacity_ || num_slots <= 0) {
        return false;
    }
    if (mapped_slots_ - num_slots < 0) {
        return false;
    }
    mapped_slots_ -= num_slots;
    capped_free_list_.SetCap(mapped_slots_);
    return true;
}

void MambaChunkAllocator::CapSlots(const std::vector<std::int32_t>& slot_ids) {
    if (!enable_dynamic_capacity_) {
        throw std::runtime_error("MambaChunkAllocator::CapSlots requires dynamic capacity mode");
    }
    for (std::int32_t id : slot_ids) {
        capped_free_list_.MarkCapped(id);
    }
}

void MambaChunkAllocator::UncapSlots(const std::vector<std::int32_t>& slot_ids) {
    if (!enable_dynamic_capacity_) {
        throw std::runtime_error("MambaChunkAllocator::UncapSlots requires dynamic capacity mode");
    }
    for (std::int32_t id : slot_ids) {
        capped_free_list_.UnmarkCapped(id);
    }
}

MambaSlot::~MambaSlot() {
    release();
}

std::int32_t MambaChunkAllocator::CappedInflightSlots() const {
    if (!enable_dynamic_capacity_) {
        return 0;
    }
    return capped_free_list_.InFlightCappedCount();
}

void MambaSlot::release() {
    if (index_ >= 0 && releaser_) {
        releaser_(index_);
        index_ = -1;
        releaser_ = {};
    }
}

}  // namespace tokenspeed
