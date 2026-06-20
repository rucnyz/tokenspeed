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

#include "budgeter/budget_agent.h"

#include <algorithm>
#include <cmath>

namespace tokenspeed {

namespace {

EvictionConfig MakeEvictionConfig(const SchedulerConfig& config) {
    EvictionConfig out;
    out.policy = EvictionConfig::ParsePolicy(config.eviction_policy);
    out.lpb_window_s = config.lpb_window_s;
    out.lpb_hit_deque_maxlen = config.lpb_hit_deque_maxlen;
    out.c_kv_alpha = config.c_kv_alpha;
    out.c_kv_beta = config.c_kv_beta;
    out.c_kv_gamma = config.c_kv_gamma;
    out.c_m = config.c_m;
    out.kv_bytes_per_page = config.kv_bytes_per_page;
    out.mamba_bytes_per_slot = config.mamba_bytes_per_slot;
    out.mamba_cache_chunk_size = config.mamba_cache_chunk_size;
    return out;
}

}  // namespace

BudgetAgent::BudgetAgent(const SchedulerConfig& config)
    : config_{config},
      cost_model_{CostModel{.eviction_config = MakeEvictionConfig(config),
                            .c_xfer_us_per_page = config.xpool_xfer_us_per_page,
                            .w_q_us = config.xpool_queue_wait_us}},
      admitter_{cost_model_} {}

void BudgetAgent::OnRequestArrival(std::int32_t prompt_tokens, const PoolSnapshot& snapshot) {
    if (!config_.enable_admitter) {
        return;
    }
    auto decision = admitter_.DecideForRequest(prompt_tokens, snapshot);
    if (decision.action == AdmitAction::kCrossFree || decision.action == AdmitAction::kCrossEvict) {
        XPoolFirePlan plan;
        plan.direction = decision.direction;
        plan.op_id = next_op_id_++;
        plan.page_ids.reserve(static_cast<std::size_t>(decision.pages_needed));
        for (std::int32_t i = 0; i < decision.pages_needed; ++i) {
            plan.page_ids.push_back(i + 1);
        }
        pending_fire_ = std::move(plan);
    }
}

std::optional<XPoolFirePlan> BudgetAgent::Tick(const PoolSnapshot& snapshot) {
    if (!config_.enable_budgeter) {
        return std::nullopt;
    }
    const auto now = std::chrono::steady_clock::now();
    const double dt = std::chrono::duration<double>(now - last_tick_).count();
    last_tick_ = now;
    if (dt <= 0.0) {
        return std::nullopt;
    }

    const double kv_util = snapshot.kv_free_pages > 0
                               ? 1.0 - static_cast<double>(snapshot.kv_free_pages) /
                                           static_cast<double>(std::max(snapshot.kv_free_pages, 1))
                               : 1.0;
    const double mamba_util = snapshot.mamba_free_slots > 0
                                  ? 1.0 - static_cast<double>(snapshot.mamba_free_slots) /
                                              static_cast<double>(std::max(snapshot.mamba_free_slots, 1))
                                  : 1.0;
    const double eta = config_.xpool_ewma_tau_s > 0.0 ? std::min(1.0, dt / config_.xpool_ewma_tau_s) : 0.1;
    ewma_pressure_kv_ = (1.0 - eta) * ewma_pressure_kv_ + eta * kv_util;
    ewma_pressure_mamba_ = (1.0 - eta) * ewma_pressure_mamba_ + eta * mamba_util;

    if (ewma_pressure_kv_ > ewma_pressure_mamba_ + config_.xpool_nb_margin &&
        snapshot.mamba_free_slots > config_.xpool_mamba_floor_slots) {
        XPoolFirePlan plan;
        plan.direction = "mamba_to_kv";
        plan.op_id = next_op_id_++;
        const std::int32_t n = config_.budgeter_pages_per_fire;
        plan.page_ids.reserve(static_cast<std::size_t>(n));
        for (std::int32_t i = 0; i < n; ++i) {
            plan.page_ids.push_back(i + 1);
        }
        pending_fire_ = plan;
        return plan;
    }
    if (ewma_pressure_mamba_ > ewma_pressure_kv_ + config_.xpool_nb_margin) {
        XPoolFirePlan plan;
        plan.direction = "kv_to_mamba";
        plan.op_id = next_op_id_++;
        const std::int32_t n = config_.budgeter_pages_per_fire;
        plan.page_ids.reserve(static_cast<std::size_t>(n));
        for (std::int32_t i = 0; i < n; ++i) {
            plan.page_ids.push_back(i + 1);
        }
        pending_fire_ = plan;
        return plan;
    }
    return std::nullopt;
}

}  // namespace tokenspeed
