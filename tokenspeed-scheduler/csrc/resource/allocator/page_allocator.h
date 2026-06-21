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
#include <vector>

#include "resource/allocator/capped_free_list.h"

namespace tokenspeed {

class OwnedPages;

class PageAllocator {
public:
    PageAllocator(std::int32_t page_size, std::int32_t total_pages, bool enable_dynamic_capacity = false);

    PageAllocator(const PageAllocator&) = delete;
    PageAllocator& operator=(const PageAllocator&) = delete;

    PageAllocator(PageAllocator&&) = delete;
    PageAllocator& operator=(PageAllocator&&) = delete;

    OwnedPages Allocate(std::int32_t num_pages);
    void Deallocate(const std::vector<std::int32_t>& pages);

    std::int32_t PageSize() const { return page_size_; };
    std::int32_t TotalPages() const { return total_pages_; }
    std::int32_t MappedPages() const { return mapped_pages_; }
    std::int32_t AvailablePages() const;
    bool DynamicCapacityEnabled() const { return enable_dynamic_capacity_; }

    std::vector<std::int32_t> Grow(std::int32_t num_pages);
    bool Shrink(std::int32_t num_pages);
    void CapPages(const std::vector<std::int32_t>& page_ids);
    void UncapPages(const std::vector<std::int32_t>& page_ids);

    // Number of capped pages still held by in-flight requests (not yet freed).
    // Returns 0 when it is safe to physically unmap the capped tail.
    std::int32_t CappedInflightPages() const;

private:
    std::int32_t page_size_{};
    std::int32_t total_pages_{};
    std::int32_t mapped_pages_{0};
    bool enable_dynamic_capacity_{false};
    std::vector<std::int32_t> free_pages_;
    CappedFreeList capped_free_list_{};
};

}  // namespace tokenspeed
