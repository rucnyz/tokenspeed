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
#include <unordered_set>
#include <vector>

namespace tokenspeed {

// Cap-barrier aware free list for HiMA-style cross-pool transfer. Capped page ids
// never appear in the allocatable free list; tail_lo tracks a contiguous capped
// suffix without materializing it.
class CappedFreeList {
public:
    static constexpr std::int32_t kNoTail = -1;

    void Reset(std::int32_t size, std::vector<std::int32_t> initial_free);

    std::optional<std::int32_t> Allocate();
    void Deallocate(std::int32_t page_id);

    void MarkCapped(std::int32_t page_id);
    void UnmarkCapped(std::int32_t page_id);
    void SetCap(std::int32_t tail_lo);

    bool IsCapped(std::int32_t page_id) const;
    std::int32_t Live() const;
    std::int32_t Available() const { return static_cast<std::int32_t>(free_ids_.size()); }
    std::int32_t NumCapped() const { return n_capped_; }
    std::int32_t Size() const { return size_; }

private:
    bool inTail(std::int32_t page_id) const;

    std::int32_t size_{0};
    std::int32_t n_capped_{0};
    std::int32_t tail_lo_{kNoTail};
    std::vector<std::int32_t> free_ids_{};
    std::unordered_set<std::int32_t> marks_{};
};

}  // namespace tokenspeed
