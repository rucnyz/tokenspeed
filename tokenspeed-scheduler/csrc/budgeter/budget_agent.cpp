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
#include <cstdlib>
#include <memory>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

namespace tokenspeed {

namespace {

// Budgeter diagnostics are routed to a dedicated file (default
// /tmp/tokenspeed_budget_tick.log, overridable via TOKENSPEED_BUDGET_LOG) so
// they survive the engine subprocess environment:
//   * Third-party code linked into the same process — notably flashinfer's
//     logging.h — calls spdlog::set_default_logger() to install its own
//     stdout sink, hijacking the global default logger.
//   * The scheduler runs in a spawned subprocess whose stdout (fd 1) AND
//     stderr (fd 2) are redirected away from the captured stream during
//     torch/CUDA init (Python logging survives only because it holds a dup of
//     the original stream).  Neither stdout nor stderr sinks reliably reach
//     the captured output.
// A file sink is immune to both issues.  Setting TOKENSPEED_BUDGET_LOG=""
// (empty string) disables diagnostics entirely for production.
bool BudgetLogEnabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("TOKENSPEED_BUDGET_LOG");
        return env == nullptr || env[0] != '\0';
    }();
    return enabled;
}

spdlog::logger& BudgetLogger() {
    static std::shared_ptr<spdlog::logger> logger = [] {
        const char* env = std::getenv("TOKENSPEED_BUDGET_LOG");
        const std::string path = (env != nullptr && env[0] != '\0')
                                     ? std::string{env}
                                     : std::string{"/tmp/tokenspeed_budget_tick.log"};
        auto sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(path, /*truncate=*/true);
        auto lg = std::make_shared<spdlog::logger>("budgeter", sink);
        lg->set_level(spdlog::level::info);
        lg->flush_on(spdlog::level::info);
        return lg;
    }();
    return *logger;
}

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

    // Utilisation = 1 - free / total.  Guard against zero total (pool not yet
    // initialised) by clamping the denominator to at least 1.
    const double kv_util = 1.0 - static_cast<double>(snapshot.kv_free_pages) /
                                     static_cast<double>(std::max(snapshot.kv_total_pages, 1));
    // Effective mamba utilisation: subtract evictable prefix-cache slots so
    // that cached-but-evictable mamba states do not inflate pressure and block
    // mamba_to_kv fires when KV demand exceeds genuine mamba demand.
    const std::int32_t mamba_used =
        std::max(0, snapshot.mamba_total_slots - snapshot.mamba_free_slots
                    - snapshot.mamba_evictable_slots);
    const double mamba_util = static_cast<double>(mamba_used) /
                              static_cast<double>(std::max(snapshot.mamba_total_slots, 1));
    const double eta = config_.xpool_ewma_tau_s > 0.0 ? std::min(1.0, dt / config_.xpool_ewma_tau_s) : 0.1;
    ewma_pressure_kv_ = (1.0 - eta) * ewma_pressure_kv_ + eta * kv_util;
    ewma_pressure_mamba_ = (1.0 - eta) * ewma_pressure_mamba_ + eta * mamba_util;

    const std::int32_t n_kv = config_.budgeter_pages_per_fire;
    // Compute how many mamba slots correspond to one fire worth of KV pages.
    std::int32_t n_mamba = 0;
    if (config_.mamba_bytes_per_slot > 0 && config_.kv_bytes_per_page > 0) {
        n_mamba = static_cast<std::int32_t>(
            static_cast<std::int64_t>(n_kv) *
            static_cast<std::int64_t>(config_.kv_bytes_per_page) /
            static_cast<std::int64_t>(config_.mamba_bytes_per_slot));
    }

    if (BudgetLogEnabled()) {
        BudgetLogger().info(
            "BudgetTick: kv_util={:.4f} mamba_util={:.4f}(eff) ewma_kv={:.4f} ewma_mamba={:.4f} "
            "kv_free={} kv_total={} mamba_free={} mamba_used={} mamba_evict={} mamba_total={} kv_headroom={}",
            kv_util, mamba_util, ewma_pressure_kv_, ewma_pressure_mamba_,
            snapshot.kv_free_pages, snapshot.kv_total_pages,
            snapshot.mamba_free_slots, mamba_used,
            snapshot.mamba_evictable_slots, snapshot.mamba_total_slots,
            snapshot.kv_headroom_pages);
    }

    if (ewma_pressure_kv_ > ewma_pressure_mamba_ + config_.xpool_nb_margin &&
        snapshot.mamba_free_slots > config_.xpool_mamba_floor_slots) {
        // Guard: only fire if KV has physical VMM headroom to receive handles.
        if (n_kv > 0 && snapshot.kv_headroom_pages < n_kv) {
            if (BudgetLogEnabled()) {
                BudgetLogger().info("BudgetTick: mamba_to_kv blocked by kv_headroom ({} < {})",
                                    snapshot.kv_headroom_pages, n_kv);
            }
            return std::nullopt;
        }
        XPoolFirePlan plan;
        plan.direction = "mamba_to_kv";
        plan.op_id = next_op_id_++;
        plan.page_ids.reserve(static_cast<std::size_t>(n_kv));
        for (std::int32_t i = 0; i < n_kv; ++i) {
            plan.page_ids.push_back(i + 1);
        }
        pending_fire_ = plan;
        return plan;
    }
    if (ewma_pressure_mamba_ > ewma_pressure_kv_ + config_.xpool_nb_margin) {
        // Guard: only fire if mamba has physical VMM headroom to receive handles.
        if (n_mamba > 0 && snapshot.mamba_headroom_slots < n_mamba) {
            return std::nullopt;
        }
        XPoolFirePlan plan;
        plan.direction = "kv_to_mamba";
        plan.op_id = next_op_id_++;
        plan.page_ids.reserve(static_cast<std::size_t>(n_kv));
        for (std::int32_t i = 0; i < n_kv; ++i) {
            plan.page_ids.push_back(i + 1);
        }
        pending_fire_ = plan;
        return plan;
    }
    return std::nullopt;
}

}  // namespace tokenspeed
