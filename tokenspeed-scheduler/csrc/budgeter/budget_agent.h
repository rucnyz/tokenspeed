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

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "budgeter/admitter.h"
#include "budgeter/cost_model.h"
#include "scheduler/types.h"
#include "resource/types.h"

namespace tokenspeed {

class Scheduler;

struct XPoolFirePlan {
    std::string direction;
    std::vector<std::int32_t> page_ids;
    cache_op_id op_id{0};
};

class BudgetAgent {
public:
    explicit BudgetAgent(const SchedulerConfig& config);

    void OnRequestArrival(std::int32_t prompt_tokens, const PoolSnapshot& snapshot);
    std::optional<XPoolFirePlan> Tick(const PoolSnapshot& snapshot);
    std::optional<XPoolFirePlan> PendingFire() const { return pending_fire_; }
    void ClearPendingFire() { pending_fire_.reset(); }

    const Admitter& GetAdmitter() const { return admitter_; }

private:
    double ewma_pressure_kv_{0.0};
    double ewma_pressure_mamba_{0.0};
    std::chrono::steady_clock::time_point last_tick_{std::chrono::steady_clock::now()};

    SchedulerConfig config_;
    CostModel cost_model_;
    Admitter admitter_;
    std::optional<XPoolFirePlan> pending_fire_{};
    cache_op_id next_op_id_{1};
};

}  // namespace tokenspeed
