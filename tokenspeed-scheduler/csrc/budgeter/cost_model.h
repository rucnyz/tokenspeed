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

#include "resource/eviction_config.h"

namespace tokenspeed {

struct CostModel {
    EvictionConfig eviction_config{};
    double c_xfer_us_per_page{70.0};
    double w_q_us{1000.0};

    double CXferUs(std::int32_t num_pages) const { return c_xfer_us_per_page * num_pages; }
    double CMigrateUs() const { return eviction_config.c_m; }
    double CEvictUs(double seq_len_tokens, std::int64_t bytes) const;
    double CRecomputeUs(double seq_len_tokens, bool is_mamba = false) const;
    double WQueueUs() const { return w_q_us; }
};

}  // namespace tokenspeed
