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

// Budgeter diagnostics are opt-in via the TOKENSPEED_BUDGET_LOG env var so
// production deployments stay quiet by default.  When the env var is set to
// a non-empty path, BudgetTick records are appended to that file.
//
// Implementation note: a file sink is required because:
//   * The scheduler runs in a spawned subprocess whose stdout/stderr are
//     redirected away from the captured stream during torch/CUDA init, so
//     stdout/stderr spdlog sinks reach nothing the operator can see.
//   * Third-party code (notably flashinfer's logging.h) hijacks the global
//     spdlog default logger, so spdlog::info() routes to a stdout sink that
//     is then lost in the redirection above.
// A dedicated named logger bound to a file sink sidesteps both issues.
bool BudgetLogEnabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("TOKENSPEED_BUDGET_LOG");
        return env != nullptr && env[0] != '\0';
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
    // Saturation gate (S2.1): when both pools are well below capacity, the
    // admitter's CrossFree / CrossEvict candidates can't beat the trivial
    // "fit-in-current-pool" path anyway, so skip the (non-trivial) decision
    // work. We use the EWMA pressure tracked by Tick() rather than the raw
    // snapshot so transient dips in a steadily-loaded pool don't repeatedly
    // re-arm the admitter.
    if (config_.xpool_saturation_low > 0.0 &&
        ewma_pressure_kv_ < config_.xpool_saturation_low &&
        ewma_pressure_mamba_ < config_.xpool_saturation_low) {
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
    } else if (decision.action == AdmitAction::kCrossMigrate) {
        // S2.6: latch a migration plan so the Python actuator can retract the
        // best victim request and free enough KV pages for the new request.
        XPoolMigratePlan plan;
        plan.pages_needed = decision.pages_needed;
        plan.op_id = next_op_id_++;
        pending_migrate_ = std::move(plan);
    } else if (decision.action == AdmitAction::kDefer) {
        // S2.3 w_paused: count explicit defer decisions so MakePoolSnapshot
        // can expose paused_count to the PressureAdapter EWMA.
        deferred_count_++;
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

    // PressureAdapter EWMA (S2.3): smooth the system-level signals with the
    // same horizon as utilisation.  Cast to double so the time-decayed
    // average has fractional resolution even when raw counts are zero/one.
    ewma_queue_   = (1.0 - eta) * ewma_queue_   + eta * static_cast<double>(snapshot.queue_len);
    ewma_retract_ = (1.0 - eta) * ewma_retract_ + eta * static_cast<double>(snapshot.retracted_count);
    ewma_paused_  = (1.0 - eta) * ewma_paused_  + eta * static_cast<double>(snapshot.paused_count);

    const std::int32_t n_kv = config_.budgeter_pages_per_fire;
    // Compute how many mamba slots correspond to one fire worth of KV pages.
    std::int32_t n_mamba = 0;
    if (config_.mamba_bytes_per_slot > 0 && config_.kv_bytes_per_page > 0) {
        n_mamba = static_cast<std::int32_t>(
            static_cast<std::int64_t>(n_kv) *
            static_cast<std::int64_t>(config_.kv_bytes_per_page) /
            static_cast<std::int64_t>(config_.mamba_bytes_per_slot));
    }

    // PressureAdapter (S2.3): blend the queue/retract/paused EWMA into the
    // direction-decision pressure values.  When all weights default to 0
    // these helpers simply return the raw EWMA so the math is unchanged.
    const double adj_kv    = AdjustedPressureKv(ewma_pressure_kv_);
    const double adj_mamba = AdjustedPressureMamba(ewma_pressure_mamba_);

    if (BudgetLogEnabled()) {
        BudgetLogger().info(
            "BudgetTick: kv_util={:.4f} mamba_util={:.4f}(eff) ewma_kv={:.4f} ewma_mamba={:.4f} "
            "adj_kv={:.4f} adj_mamba={:.4f} ewma_q={:.2f} ewma_r={:.2f} ewma_p={:.2f} "
            "kv_free={} kv_total={} mamba_free={} mamba_used={} mamba_evict={} mamba_total={} "
            "queue={} retract={} paused={} kv_headroom={}",
            kv_util, mamba_util, ewma_pressure_kv_, ewma_pressure_mamba_,
            adj_kv, adj_mamba,
            ewma_queue_, ewma_retract_, ewma_paused_,
            snapshot.kv_free_pages, snapshot.kv_total_pages,
            snapshot.mamba_free_slots, mamba_used,
            snapshot.mamba_evictable_slots, snapshot.mamba_total_slots,
            snapshot.queue_len, snapshot.retracted_count, snapshot.paused_count,
            snapshot.kv_headroom_pages);
    }

    // Saturation gate (S2.1): only run the fire-decision branches when at
    // least one pool is approaching saturation. EWMA is still updated above
    // so rising load is detected the moment either side crosses the gate.
    // The gate uses the *adjusted* pressure so a heavy queue/retract burst
    // can lift the budgeter out of the low-saturation no-op regime even
    // before raw utilisation EWMA climbs.
    if (config_.xpool_saturation_low > 0.0 &&
        adj_kv < config_.xpool_saturation_low &&
        adj_mamba < config_.xpool_saturation_low) {
        if (BudgetLogEnabled()) {
            BudgetLogger().info(
                "BudgetTick: gated by xpool_saturation_low={:.2f} "
                "(adj_kv={:.3f} adj_mamba={:.3f})",
                config_.xpool_saturation_low, adj_kv, adj_mamba);
        }
        deferred_count_ = 0;
        return std::nullopt;
    }

    if (adj_kv > adj_mamba + config_.xpool_nb_margin &&
        snapshot.mamba_free_slots > config_.xpool_mamba_floor_slots) {
        // Guard: only fire if KV has physical VMM headroom to receive handles.
        if (n_kv > 0 && snapshot.kv_headroom_pages < n_kv) {
            if (BudgetLogEnabled()) {
                BudgetLogger().info("BudgetTick: mamba_to_kv blocked by kv_headroom ({} < {})",
                                    snapshot.kv_headroom_pages, n_kv);
            }
            deferred_count_ = 0;
            return std::nullopt;
        }
        // S2.2 reverse-direction cooldown.
        if (ReverseDirectionGated("mamba_to_kv", now)) {
            if (BudgetLogEnabled()) {
                BudgetLogger().info(
                    "BudgetTick: mamba_to_kv suppressed by reverse_cooldown (last_fire={}, within {:.2f}s)",
                    last_fire_direction_, config_.xpool_reverse_cooldown_s);
            }
            deferred_count_ = 0;
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
        deferred_count_ = 0;
        return plan;
    }
    if (adj_mamba > adj_kv + config_.xpool_nb_margin) {
        // Guard: only fire if mamba has physical VMM headroom to receive handles.
        if (n_mamba > 0 && snapshot.mamba_headroom_slots < n_mamba) {
            deferred_count_ = 0;
            return std::nullopt;
        }
        // S2.2 reverse-direction cooldown.
        if (ReverseDirectionGated("kv_to_mamba", now)) {
            if (BudgetLogEnabled()) {
                BudgetLogger().info(
                    "BudgetTick: kv_to_mamba suppressed by reverse_cooldown (last_fire={}, within {:.2f}s)",
                    last_fire_direction_, config_.xpool_reverse_cooldown_s);
            }
            deferred_count_ = 0;
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
        deferred_count_ = 0;
        return plan;
    }
    // S2.3 w_paused: reset deferred_count_ at the end of each tick so the
    // accumulation window covers exactly one inter-tick interval.
    deferred_count_ = 0;
    return std::nullopt;
}

void BudgetAgent::OnFireCommitted(const std::string& direction) {
    // Only physical commits (Scheduler::ApplyXPoolFire) call this. Cancelled
    // fires (Scheduler::CancelXPoolFire) intentionally bypass this hook so the
    // cooldown gate never punishes a no-op cancel.
    if (direction != "kv_to_mamba" && direction != "mamba_to_kv") {
        return;
    }
    last_fire_direction_ = direction;
    last_fire_time_ = std::chrono::steady_clock::now();
}

double BudgetAgent::AdjustedPressureKv(double base_pressure) const {
    // PressureAdapter (S2.3).  Skip the math when both weights are zero so
    // the no-config-change codepath stays branch-light.
    if (config_.xpool_w_queue <= 0.0 && config_.xpool_w_retract <= 0.0) {
        return base_pressure;
    }
    const double queue_ref = static_cast<double>(std::max(
        1, config_.xpool_queue_ref > 0
               ? config_.xpool_queue_ref
               : (config_.max_batch_size > 0 ? config_.max_batch_size / 2 : 1)));
    const double retract_ref = static_cast<double>(std::max(
        1, config_.xpool_retract_ref > 0
               ? config_.xpool_retract_ref
               : (config_.max_batch_size > 0 ? config_.max_batch_size / 4 : 1)));
    const double q_norm = std::clamp(ewma_queue_ / queue_ref, 0.0, 1.0);
    const double r_norm = std::clamp(ewma_retract_ / retract_ref, 0.0, 1.0);
    const double adjusted = base_pressure
                          + config_.xpool_w_queue * q_norm
                          + config_.xpool_w_retract * r_norm;
    return std::clamp(adjusted, 0.0, 1.0);
}

double BudgetAgent::AdjustedPressureMamba(double base_pressure) const {
    // PressureAdapter (S2.3).  Currently only paused contributes; the
    // queue/retract signals are KV-side indicators.
    if (config_.xpool_w_paused <= 0.0) {
        return base_pressure;
    }
    const double paused_ref = static_cast<double>(std::max(
        1, config_.xpool_paused_ref > 0
               ? config_.xpool_paused_ref
               : (config_.max_batch_size > 0 ? config_.max_batch_size / 4 : 1)));
    const double p_norm = std::clamp(ewma_paused_ / paused_ref, 0.0, 1.0);
    const double adjusted = base_pressure + config_.xpool_w_paused * p_norm;
    return std::clamp(adjusted, 0.0, 1.0);
}

bool BudgetAgent::ReverseDirectionGated(
    const std::string& direction,
    std::chrono::steady_clock::time_point now) const {
    if (config_.xpool_reverse_cooldown_s <= 0.0) {
        return false;
    }
    if (last_fire_direction_.empty()) {
        return false;
    }
    if (last_fire_direction_ == direction) {
        // Same-direction fires accumulate capacity transfer; never gated.
        return false;
    }
    const double elapsed =
        std::chrono::duration<double>(now - last_fire_time_).count();
    return elapsed < config_.xpool_reverse_cooldown_s;
}

}  // namespace tokenspeed
