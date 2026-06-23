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

// S2.6: plan emitted by the budgeter when the admitter selects kCrossMigrate.
// The Python actuator receives this via pending_xpool_migrate(), retracts the
// best victim request (BestMigrateCandidate), and calls apply_xpool_migrate()
// to commit the operation and increment the migration counter.
struct XPoolMigratePlan {
    std::int32_t pages_needed{0};
    cache_op_id op_id{0};
};

class BudgetAgent {
public:
    explicit BudgetAgent(const SchedulerConfig& config);

    void OnRequestArrival(std::int32_t prompt_tokens, const PoolSnapshot& snapshot);
    std::optional<XPoolFirePlan> Tick(const PoolSnapshot& snapshot);
    std::optional<XPoolFirePlan> PendingFire() const { return pending_fire_; }
    void ClearPendingFire() { pending_fire_.reset(); }

    // S2.6 migration plan accessors.
    std::optional<XPoolMigratePlan> PendingMigrate() const { return pending_migrate_; }
    void ClearPendingMigrate() { pending_migrate_.reset(); }
    void IncrCommittedMigrate() { committed_migrate_count_++; }
    std::int32_t CommittedMigrateCount() const { return committed_migrate_count_; }

    // S2.3 w_paused: number of kDefer decisions emitted since last Tick().
    // Scheduler::MakePoolSnapshot() reads this to populate paused_count.
    std::int32_t DeferredCount() const { return deferred_count_; }

    // S2.2 reverse-direction cooldown. Called from Scheduler::ApplyXPoolFire
    // AFTER the physical Grow/Shrink succeeds, so cancelled fires (which take
    // the CancelXPoolFire / ClearPendingFire path) do not arm the cooldown.
    // Subsequent Tick() calls within xpool_reverse_cooldown_s will refuse to
    // emit a plan in the OPPOSITE direction; same-direction plans are still
    // allowed so a sustained pressure imbalance can keep transferring capacity.
    void OnFireCommitted(const std::string& direction);

    // Test introspection. Returns the direction recorded by the most recent
    // OnFireCommitted call, or empty string if no fire has committed yet.
    const std::string& LastFireDirection() const { return last_fire_direction_; }

    // PressureAdapter introspection (S2.3): the EWMA-smoothed adjusted
    // pressures last computed by Tick().  Useful for unit tests that want
    // to verify the queue/retract weighting actually moves the comparison.
    double EwmaPressureKv() const { return ewma_pressure_kv_; }
    double EwmaPressureMamba() const { return ewma_pressure_mamba_; }
    double EwmaQueue() const { return ewma_queue_; }
    double EwmaRetract() const { return ewma_retract_; }
    double EwmaPaused() const { return ewma_paused_; }

    const Admitter& GetAdmitter() const { return admitter_; }

private:
    // True iff a fire in `direction` would be in the opposite direction of
    // the last committed fire and we're still within the cooldown window.
    // Returns false if cooldown is disabled (xpool_reverse_cooldown_s <= 0)
    // or if no prior fire has committed.
    bool ReverseDirectionGated(const std::string& direction,
                               std::chrono::steady_clock::time_point now) const;

    // PressureAdapter (S2.3): apply the weighted queue/retract/paused
    // signals on top of the raw utilisation EWMA, returning the adjusted
    // pressure value clamped to [0, 1].  When all weights are 0 this
    // simply returns *base_pressure*.
    double AdjustedPressureKv(double base_pressure) const;
    double AdjustedPressureMamba(double base_pressure) const;

    double ewma_pressure_kv_{0.0};
    double ewma_pressure_mamba_{0.0};
    // PressureAdapter EWMA state.  Updated each Tick using the same
    // time-decayed weight as the utilisation EWMA so the signals share a
    // smoothing horizon.
    double ewma_queue_{0.0};
    double ewma_retract_{0.0};
    double ewma_paused_{0.0};
    std::chrono::steady_clock::time_point last_tick_{std::chrono::steady_clock::now()};

    // S2.2 reverse-direction cooldown state. Updated only on physical commit
    // via OnFireCommitted (Scheduler::ApplyXPoolFire), never on cancel.
    std::string last_fire_direction_{};
    std::chrono::steady_clock::time_point last_fire_time_{};

    SchedulerConfig config_;
    CostModel cost_model_;
    Admitter admitter_;
    std::optional<XPoolFirePlan> pending_fire_{};
    // S2.6: migration plan latched by OnRequestArrival when admitter decides
    // kCrossMigrate.  Cleared by ApplyXPoolMigrate / CancelXPoolMigrate.
    std::optional<XPoolMigratePlan> pending_migrate_{};
    std::int32_t committed_migrate_count_{0};
    // S2.3 w_paused: accumulated kDefer count since last Tick().  Reset at
    // the end of each Tick() call so MakePoolSnapshot (called just before
    // Tick) sees the count for the preceding inter-tick interval.
    std::int32_t deferred_count_{0};
    cache_op_id next_op_id_{1};
};

}  // namespace tokenspeed
