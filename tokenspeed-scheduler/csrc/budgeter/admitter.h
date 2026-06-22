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
#include <string>

#include "budgeter/cost_model.h"

namespace tokenspeed {

enum class AdmitAction {
    kOwnFree,
    kOwnEvict,
    kOwnMigrate,
    kCrossFree,
    kCrossEvict,
    kCrossMigrate,
    kDefer,
};

struct AdmitDecision {
    AdmitAction action{AdmitAction::kDefer};
    double cost_us{0.0};
    std::string direction;  // "kv_to_mamba" | "mamba_to_kv" | ""
    std::int32_t pages_needed{0};
};

struct PoolSnapshot {
    std::int32_t kv_free_pages{0};
    std::int32_t kv_total_pages{0};
    std::int32_t kv_evictable_pages{0};
    std::int32_t mamba_free_slots{0};
    std::int32_t mamba_total_slots{0};
    std::int32_t mamba_evictable_slots{0};
    std::int32_t queue_len{0};

    // Physical arena headroom: slots/pages that have reserved VMM VA but are
    // not yet mapped.  The budgeter checks these before issuing a fire so it
    // does not generate plans that the Python actuator would immediately cancel.
    std::int32_t kv_headroom_pages{0};
    std::int32_t mamba_headroom_slots{0};
};

class Admitter {
public:
    explicit Admitter(CostModel cost_model);

    AdmitDecision DecideForRequest(std::int32_t prompt_tokens, const PoolSnapshot& snapshot,
                                   std::int32_t mamba_need_slots = 2) const;

private:
    AdmitDecision decide(std::int32_t x_tokens, std::int32_t x_eff, const PoolSnapshot& snapshot,
                         bool cross, const std::string& direction) const;

    CostModel cost_model_;
    std::int32_t pages_per_fire_{64};
};

}  // namespace tokenspeed
