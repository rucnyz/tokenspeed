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
#include <string>

namespace tokenspeed {

enum class EvictionPolicy {
    kLru,
    kLpb,
};

struct EvictionConfig {
    EvictionPolicy policy{EvictionPolicy::kLru};
    double lpb_window_s{60.0};
    std::int32_t lpb_hit_deque_maxlen{4096};
    // Quadratic re-prefill cost c_KV(L) = alpha*L^2 + beta*L + gamma (microseconds).
    double c_kv_alpha{1.02e-7};
    double c_kv_beta{0.0246};
    double c_kv_gamma{5.97};
    // Single-curve design: hybrid miss folds into c_KV; c_M defaults to 0.
    double c_m{0.0};
    std::int64_t kv_bytes_per_page{0};
    std::int64_t mamba_bytes_per_slot{0};
    std::int32_t mamba_cache_chunk_size{64};

    static EvictionPolicy ParsePolicy(const std::string& name);

    double RecomputeCostUs(double seq_len_tokens, bool is_mamba) const;
};

}  // namespace tokenspeed
