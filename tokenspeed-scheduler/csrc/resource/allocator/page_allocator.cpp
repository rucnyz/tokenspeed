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

#include "resource/allocator/owned_pages.h"
#include "resource/allocator/page_allocator.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <utility>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ranges.h>

namespace tokenspeed {

PageAllocator::PageAllocator(std::int32_t page_size, std::int32_t total_pages, bool enable_dynamic_capacity)
    : page_size_(page_size),
      total_pages_(total_pages),
      mapped_pages_(enable_dynamic_capacity ? 0 : (total_pages > 0 ? total_pages - 1 : 0)),
      enable_dynamic_capacity_(enable_dynamic_capacity) {
    if (enable_dynamic_capacity_) {
        // Dynamic mode: start with no pages available; caller uses Grow() to expand.
        capped_free_list_.Reset(total_pages_, {});
    } else {
        free_pages_.reserve(static_cast<std::size_t>(total_pages_));
        for (std::int32_t i = 1; i < total_pages_; ++i) {
            free_pages_.push_back(i);
        }
    }
}

std::int32_t PageAllocator::AvailablePages() const {
    if (enable_dynamic_capacity_) {
        return capped_free_list_.Available();
    }
    return static_cast<std::int32_t>(free_pages_.size());
}

OwnedPages PageAllocator::Allocate(std::int32_t num_pages) {
    if (num_pages <= 0) {
        return {};
    }
    if (enable_dynamic_capacity_) {
        std::vector<std::int32_t> pages;
        pages.reserve(static_cast<std::size_t>(num_pages));
        for (std::int32_t i = 0; i < num_pages; ++i) {
            auto page = capped_free_list_.Allocate();
            if (!page.has_value()) {
                return {};
            }
            pages.push_back(*page);
        }
        return OwnedPages{this, std::move(pages)};
    }
    if (static_cast<std::size_t>(num_pages) > free_pages_.size()) {
        return {};
    }
    std::vector<std::int32_t> pages;
    if (std::getenv("DEBUG_MEM")) {
        spdlog::debug("Free Pages Before Allocate: {}", free_pages_.size());
    }
    pages.reserve(static_cast<std::size_t>(num_pages));
    for (std::int32_t i = 0; i < num_pages; ++i) {
        pages.push_back(free_pages_.back());
        free_pages_.pop_back();
    }
    if (std::getenv("DEBUG_MEM")) {
        spdlog::debug("Free Pages After Allocate: {}", free_pages_.size());
        spdlog::debug("Allocated pages: [{}]", fmt::join(pages, ", "));
    }
    return OwnedPages{this, std::move(pages)};
}

void PageAllocator::Deallocate(const std::vector<std::int32_t>& pages) {
    if (enable_dynamic_capacity_) {
        for (std::int32_t page : pages) {
            capped_free_list_.Deallocate(page);
        }
        return;
    }
    if (std::getenv("DEBUG_MEM")) {
        spdlog::debug("Pages to Deallocate: [{}]", fmt::join(pages, ", "));
        spdlog::debug("Free Pages Before Deallocate: {}", free_pages_.size());
    }
    free_pages_.insert(free_pages_.end(), pages.begin(), pages.end());
    if (std::getenv("DEBUG_MEM")) {
        spdlog::debug("Free Pages After Deallocate: {}", free_pages_.size());
    }
}

std::vector<std::int32_t> PageAllocator::Grow(std::int32_t num_pages) {
    if (!enable_dynamic_capacity_ || num_pages <= 0) {
        return {};
    }
    if (mapped_pages_ + num_pages > total_pages_ - 1) {
        throw std::runtime_error("PageAllocator::Grow: exceeds reserved capacity");
    }
    const std::int32_t old_mapped = mapped_pages_;
    mapped_pages_ += num_pages;
    // Record how many pages beyond this first Grow are "headroom" — they are
    // tail-capped (by SetCap below) but have never been allocated to any request.
    // CappedInflightPages() subtracts this constant so the drain gate works.
    if (first_grow_) {
        headroom_pages_ = (total_pages_ - 1) - mapped_pages_;
        first_grow_ = false;
    }
    // The newly mapped pages may currently sit inside a tail-cap region left by
    // a prior Shrink (pages lent to the peer pool). Raise the cap barrier above
    // them first so they are no longer considered capped, then return them to
    // the free list. Always re-asserting the barrier at mapped_pages_ + 1 keeps
    // Grow correct for both the initial baseline restore (no prior tail) and
    // reclaiming previously lent-out pages.
    capped_free_list_.SetCap(mapped_pages_ + 1);
    std::vector<std::int32_t> grown;
    grown.reserve(static_cast<std::size_t>(num_pages));
    for (std::int32_t page_id = old_mapped + 1; page_id <= mapped_pages_; ++page_id) {
        capped_free_list_.Deallocate(page_id);
        grown.push_back(page_id);
    }
    return grown;
}

bool PageAllocator::Shrink(std::int32_t num_pages) {
    if (!enable_dynamic_capacity_ || num_pages <= 0) {
        return false;
    }
    if (mapped_pages_ - num_pages < 1) {
        return false;
    }
    mapped_pages_ -= num_pages;
    capped_free_list_.SetCap(mapped_pages_ + 1);
    return true;
}

void PageAllocator::CapPages(const std::vector<std::int32_t>& page_ids) {
    if (!enable_dynamic_capacity_) {
        throw std::runtime_error("PageAllocator::CapPages requires dynamic capacity mode");
    }
    for (std::int32_t page_id : page_ids) {
        capped_free_list_.MarkCapped(page_id);
    }
}

void PageAllocator::UncapPages(const std::vector<std::int32_t>& page_ids) {
    if (!enable_dynamic_capacity_) {
        throw std::runtime_error("PageAllocator::UncapPages requires dynamic capacity mode");
    }
    for (std::int32_t page_id : page_ids) {
        capped_free_list_.UnmarkCapped(page_id);
    }
}

std::int32_t PageAllocator::CappedInflightPages() const {
    if (!enable_dynamic_capacity_) {
        return 0;
    }
    // headroom_pages_ are tail-capped after the initial Grow but were never
    // allocated to any request.  They must not be counted as in-flight.
    const std::int32_t raw = capped_free_list_.InFlightCappedCount();
    return std::max(0, raw - headroom_pages_);
}

}  // namespace tokenspeed
