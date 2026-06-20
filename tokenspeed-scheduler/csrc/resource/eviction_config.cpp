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

#include "resource/eviction_config.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace tokenspeed {

EvictionPolicy EvictionConfig::ParsePolicy(const std::string& name) {
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lower == "lru") {
        return EvictionPolicy::kLru;
    }
    if (lower == "lpb") {
        return EvictionPolicy::kLpb;
    }
    throw std::invalid_argument("EvictionConfig: unknown eviction policy '" + name + "'");
}

double EvictionConfig::RecomputeCostUs(double seq_len_tokens, bool is_mamba) const {
    if (is_mamba) {
        return c_m;
    }
    const double l = seq_len_tokens;
    return c_kv_alpha * l * l + c_kv_beta * l + c_kv_gamma;
}

}  // namespace tokenspeed
