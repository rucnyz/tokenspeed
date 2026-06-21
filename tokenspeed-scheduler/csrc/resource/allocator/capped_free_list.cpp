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

#include "resource/allocator/capped_free_list.h"

#include <algorithm>
#include <stdexcept>

namespace tokenspeed {

void CappedFreeList::Reset(std::int32_t size, std::vector<std::int32_t> initial_free) {
    size_ = size;
    n_capped_ = 0;
    tail_lo_ = kNoTail;
    marks_.clear();
    capped_drained_.clear();
    free_ids_ = std::move(initial_free);
    std::sort(free_ids_.begin(), free_ids_.end());
}

bool CappedFreeList::inTail(std::int32_t page_id) const {
    return tail_lo_ != kNoTail && page_id >= tail_lo_ && page_id < size_;
}

bool CappedFreeList::IsCapped(std::int32_t page_id) const {
    if (page_id < 0 || page_id >= size_) {
        return false;
    }
    if (inTail(page_id)) {
        return true;
    }
    return marks_.count(page_id) > 0;
}

std::int32_t CappedFreeList::Live() const {
    return size_ - n_capped_;
}

void CappedFreeList::MarkCapped(std::int32_t page_id) {
    if (IsCapped(page_id)) {
        return;
    }
    auto it = std::find(free_ids_.begin(), free_ids_.end(), page_id);
    if (it != free_ids_.end()) {
        // Page was free; removing it makes it immediately drained.
        free_ids_.erase(it);
        capped_drained_.insert(page_id);
    }
    marks_.insert(page_id);
    ++n_capped_;
}

void CappedFreeList::UnmarkCapped(std::int32_t page_id) {
    if (!IsCapped(page_id)) {
        return;
    }
    if (inTail(page_id)) {
        throw std::runtime_error("CappedFreeList: cannot unmark tail-capped page individually");
    }
    marks_.erase(page_id);
    --n_capped_;
    // Remove from drained set; the page is live again.
    capped_drained_.erase(page_id);
    free_ids_.push_back(page_id);
    std::sort(free_ids_.begin(), free_ids_.end());
}

void CappedFreeList::SetCap(std::int32_t tail_lo) {
    if (tail_lo == kNoTail) {
        if (tail_lo_ != kNoTail) {
            // Entire tail cap lifted: remove those pages from the drain set.
            for (std::int32_t id = tail_lo_; id < size_; ++id) {
                capped_drained_.erase(id);
            }
            n_capped_ -= (size_ - tail_lo_);
            tail_lo_ = kNoTail;
        }
        return;
    }
    if (tail_lo < 0 || tail_lo > size_) {
        throw std::invalid_argument("CappedFreeList::SetCap: tail_lo out of range");
    }
    if (tail_lo_ != kNoTail) {
        if (tail_lo > tail_lo_) {
            // Cap raised (Grow): pages in [old_tail_lo, tail_lo) become live again.
            for (std::int32_t id = tail_lo_; id < tail_lo; ++id) {
                capped_drained_.erase(id);
            }
        }
        n_capped_ -= (size_ - tail_lo_);
    }
    tail_lo_ = tail_lo;
    n_capped_ += (size_ - tail_lo_);
    // Remove newly capped free pages from free_ids_ and mark them drained.
    free_ids_.erase(
        std::remove_if(free_ids_.begin(), free_ids_.end(),
                       [this](std::int32_t id) {
                           if (inTail(id) || IsCapped(id)) {
                               capped_drained_.insert(id);
                               return true;
                           }
                           return false;
                       }),
        free_ids_.end());
}

std::int32_t CappedFreeList::InFlightCappedCount() const {
    // In-flight = capped but not yet freed via Deallocate.
    return n_capped_ - static_cast<std::int32_t>(capped_drained_.size());
}

std::optional<std::int32_t> CappedFreeList::Allocate() {
    if (free_ids_.empty()) {
        return std::nullopt;
    }
    for (auto it = free_ids_.begin(); it != free_ids_.end(); ++it) {
        if (!IsCapped(*it)) {
            std::int32_t page_id = *it;
            free_ids_.erase(it);
            return page_id;
        }
    }
    return std::nullopt;
}

void CappedFreeList::Deallocate(std::int32_t page_id) {
    if (page_id < 0 || page_id >= size_) {
        throw std::out_of_range("CappedFreeList::Deallocate: page_id out of range");
    }
    if (IsCapped(page_id)) {
        // Page is capped (lent to peer pool via Shrink).  Mark it drained so
        // InFlightCappedCount() eventually reaches zero, enabling safe unmap.
        // Do NOT return it to the free list; it will be reclaimed by Grow.
        capped_drained_.insert(page_id);
        return;
    }
    free_ids_.push_back(page_id);
    std::sort(free_ids_.begin(), free_ids_.end());
}

}  // namespace tokenspeed
