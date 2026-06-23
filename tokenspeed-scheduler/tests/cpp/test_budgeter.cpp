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

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

#include "budgeter/budget_agent.h"
#include "scheduler/types.h"

namespace tokenspeed::test {

namespace {

// Build a budgeter config tuned for deterministic single-tick decisions:
// a near-zero EWMA time constant forces eta -> 1 so the smoothed pressure
// equals the instantaneous utilisation after a single tick.
SchedulerConfig MakeBudgeterConfig() {
    SchedulerConfig config;
    config.enable_budgeter = true;
    config.xpool_ewma_tau_s = 1e-9;  // eta = min(1, dt/tau) == 1
    config.xpool_nb_margin = 0.1;
    config.xpool_mamba_floor_slots = 0;
    config.budgeter_pages_per_fire = 64;
    // Most existing tests intentionally exercise far-from-saturation
    // snapshots (e.g. kv_util=0 vs mamba_util=1). Disable the S2.1
    // saturation gate by default so those tests still see fires. A
    // dedicated test below exercises the gate explicitly.
    config.xpool_saturation_low = 0.0;
    return config;
}

// Ensure dt > 0 between construction and the first Tick.
void TinySleep() {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
}

}  // namespace

TEST(BudgeterTest, DisabledBudgeterNeverFires) {
    SchedulerConfig config = MakeBudgeterConfig();
    config.enable_budgeter = false;
    BudgetAgent agent(config);

    TinySleep();
    PoolSnapshot snap{.kv_free_pages = 0, .mamba_free_slots = 100};
    EXPECT_FALSE(agent.Tick(snap).has_value());
}

// Helper to build a snapshot pre-populated with the fields that
// MakePoolSnapshot fills in production (totals + arena headroom).  Callers
// only need to override free counts / evictable / etc. to express each
// test's intent.  Defaults give both pools ample physical headroom so the
// budgeter's headroom guards never block.
PoolSnapshot MakeBaseSnapshot() {
    PoolSnapshot s;
    s.kv_total_pages = 100;
    s.mamba_total_slots = 100;
    s.kv_headroom_pages = 1000;     // far exceeds budgeter_pages_per_fire
    s.mamba_headroom_slots = 1000;
    return s;
}

TEST(BudgeterTest, KvPressureFiresMambaToKv) {
    BudgetAgent agent(MakeBudgeterConfig());

    TinySleep();
    // KV exhausted (util=1), mamba has plenty of free slots (util=0).
    PoolSnapshot snap = MakeBaseSnapshot();
    snap.kv_free_pages = 0;             // kv_util = 100%
    snap.mamba_free_slots = 100;        // mamba_util(eff) = 0%
    auto plan = agent.Tick(snap);

    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->direction, "mamba_to_kv");
    EXPECT_EQ(plan->page_ids.size(), 64u);
    EXPECT_GT(plan->op_id, 0);
}

TEST(BudgeterTest, MambaPressureFiresKvToMamba) {
    BudgetAgent agent(MakeBudgeterConfig());

    TinySleep();
    // Mamba exhausted (util=1), KV has free pages (util=0).
    PoolSnapshot snap = MakeBaseSnapshot();
    snap.kv_free_pages = 100;           // kv_util = 0%
    snap.mamba_free_slots = 0;          // mamba_util(eff) = 100%
    auto plan = agent.Tick(snap);

    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->direction, "kv_to_mamba");
    EXPECT_EQ(plan->page_ids.size(), 64u);
}

TEST(BudgeterTest, BalancedPoolsDoNotFire) {
    BudgetAgent agent(MakeBudgeterConfig());

    TinySleep();
    // Both pools at 50% utilisation -> equal pressure -> no transfer.
    PoolSnapshot snap = MakeBaseSnapshot();
    snap.kv_free_pages = 50;
    snap.mamba_free_slots = 50;
    EXPECT_FALSE(agent.Tick(snap).has_value());
}

TEST(BudgeterTest, MambaFloorBlocksMambaToKvFire) {
    SchedulerConfig config = MakeBudgeterConfig();
    config.xpool_mamba_floor_slots = 50;  // keep at least 50 mamba slots
    BudgetAgent agent(config);

    TinySleep();
    // KV exhausted and mamba has free slots, but below the protective floor:
    // the budgeter must not steal mamba capacity.
    PoolSnapshot snap = MakeBaseSnapshot();
    snap.kv_free_pages = 0;
    snap.mamba_free_slots = 10;          // below xpool_mamba_floor_slots=50
    EXPECT_FALSE(agent.Tick(snap).has_value());
}

TEST(BudgeterTest, MambaEvictableSlotsAreSubtractedFromPressure) {
    // Regression for the HiMA Phase 2 stress test: when mamba's prefix-cache
    // is full of evictable states, those slots should NOT inflate mamba
    // pressure and block a mamba_to_kv fire when KV demand is present.
    //
    // Without the evictable subtraction, mamba_util = (total-free)/total
    // counts cached states as "used", so a snapshot like (free=10, total=100,
    // evict=60) reports 90% mamba util and easily beats any modest kv_util,
    // preventing mamba_to_kv from ever firing in hybrid workloads.
    BudgetAgent agent(MakeBudgeterConfig());

    TinySleep();
    PoolSnapshot snap = MakeBaseSnapshot();
    snap.kv_total_pages = 1000;
    snap.kv_free_pages = 590;              // kv_util = 41%
    snap.mamba_free_slots = 10;
    snap.mamba_evictable_slots = 60;       // active = 100-10-60 = 30, eff = 30%
    // 41% > 30% + 10% margin -> mamba_to_kv must fire.
    auto plan = agent.Tick(snap);

    ASSERT_TRUE(plan.has_value())
        << "mamba_to_kv must fire when kv_util > effective mamba_util";
    EXPECT_EQ(plan->direction, "mamba_to_kv");
}

TEST(BudgeterTest, KvHeadroomGuardBlocksMambaToKvFire) {
    // Even with KV pressure exceeding mamba pressure, the budgeter must not
    // emit a mamba_to_kv plan when the KV arena cannot physically absorb the
    // returned handles (headroom_pages < budgeter_pages_per_fire).  Without
    // this guard, Python rejects the fire on every tick and we burn budget
    // ticks in a cancel loop.
    SchedulerConfig config = MakeBudgeterConfig();
    config.budgeter_pages_per_fire = 64;
    BudgetAgent agent(config);

    TinySleep();
    PoolSnapshot snap = MakeBaseSnapshot();
    snap.kv_free_pages = 0;                // kv_util = 100%
    snap.mamba_free_slots = 100;           // mamba_util = 0%
    snap.kv_headroom_pages = 16;           // < 64 -> must block
    EXPECT_FALSE(agent.Tick(snap).has_value())
        << "mamba_to_kv must be suppressed when KV arena lacks headroom";
}

TEST(BudgeterTest, MambaHeadroomGuardBlocksKvToMambaFire) {
    // Symmetric guard for the opposite direction.
    SchedulerConfig config = MakeBudgeterConfig();
    config.budgeter_pages_per_fire = 64;
    config.kv_bytes_per_page = 1024;
    config.mamba_bytes_per_slot = 1024;    // n_mamba == n_kv == 64
    BudgetAgent agent(config);

    TinySleep();
    PoolSnapshot snap = MakeBaseSnapshot();
    snap.kv_free_pages = 100;              // kv_util = 0%
    snap.mamba_free_slots = 0;             // mamba_util = 100%
    snap.mamba_headroom_slots = 16;        // < 64 -> must block
    EXPECT_FALSE(agent.Tick(snap).has_value())
        << "kv_to_mamba must be suppressed when mamba arena lacks headroom";
}

TEST(BudgeterTest, SaturationGateSuppressesFireWhenBothPoolsLowPressure) {
    // S2.1: With xpool_saturation_low > both pools' EWMA pressure, the
    // budgeter must skip the fire branches even when the inter-pool delta
    // exceeds xpool_nb_margin. This is the explicit "no fire when neither
    // pool is anywhere near full" guard.
    SchedulerConfig config = MakeBudgeterConfig();
    config.xpool_saturation_low = 0.5;
    BudgetAgent agent(config);

    TinySleep();
    PoolSnapshot snap = MakeBaseSnapshot();
    // KV at 30% util, mamba at 0% util. Delta (0.30) >> margin (0.1) so
    // the legacy logic would emit mamba_to_kv. With the saturation gate
    // we want this suppressed because no pool is anywhere near full.
    snap.kv_free_pages = 70;        // kv_util = 30%
    snap.mamba_free_slots = 100;    // mamba_util(eff) = 0%
    EXPECT_FALSE(agent.Tick(snap).has_value())
        << "fire must be suppressed when both pools are below xpool_saturation_low";
}

TEST(BudgeterTest, SaturationGateAllowsFireWhenOneSideSaturates) {
    // Symmetric: once at least one side crosses the saturation threshold,
    // the normal fire-decision branches run again.
    SchedulerConfig config = MakeBudgeterConfig();
    config.xpool_saturation_low = 0.5;
    BudgetAgent agent(config);

    TinySleep();
    PoolSnapshot snap = MakeBaseSnapshot();
    snap.kv_free_pages = 0;             // kv_util = 100% (>> threshold)
    snap.mamba_free_slots = 100;        // mamba_util(eff) = 0%
    auto plan = agent.Tick(snap);
    ASSERT_TRUE(plan.has_value())
        << "fire must resume once one pool crosses the saturation threshold";
    EXPECT_EQ(plan->direction, "mamba_to_kv");
}

TEST(BudgeterTest, SaturationGateDisabledByZero) {
    // xpool_saturation_low = 0.0 must restore the legacy "always run the
    // fire path" behavior so existing deployments aren't surprised.
    SchedulerConfig config = MakeBudgeterConfig();
    config.xpool_saturation_low = 0.0;
    BudgetAgent agent(config);

    TinySleep();
    PoolSnapshot snap = MakeBaseSnapshot();
    snap.kv_free_pages = 70;        // kv_util = 30%
    snap.mamba_free_slots = 100;    // mamba_util(eff) = 0%
    // With gate disabled and util delta 0.30 > margin 0.10, must fire.
    auto plan = agent.Tick(snap);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->direction, "mamba_to_kv");
}

// =====================================================================
// S2.2 reverse-direction cooldown tests
// =====================================================================
//
// After a fire commits in direction D, the budgeter must suppress any plan
// in the OPPOSITE direction for xpool_reverse_cooldown_s seconds. Same-
// direction plans and disabled cooldown (0.0) must remain unaffected. The
// hook point is BudgetAgent::OnFireCommitted, called from
// Scheduler::ApplyXPoolFire after physical Grow/Shrink succeed.

TEST(BudgeterTest, ReverseCooldownSuppressesOppositeDirection) {
    SchedulerConfig config = MakeBudgeterConfig();
    config.xpool_reverse_cooldown_s = 1.0;  // 1-second window
    BudgetAgent agent(config);

    // Step 1: a kv_to_mamba fire commits.
    agent.OnFireCommitted("kv_to_mamba");
    EXPECT_EQ(agent.LastFireDirection(), "kv_to_mamba");

    // Step 2: pressure flips. Without the cooldown this snapshot would
    // produce a mamba_to_kv plan (kv exhausted, mamba free), but the
    // gate must suppress it because we're still inside the 1s window.
    TinySleep();
    PoolSnapshot snap = MakeBaseSnapshot();
    snap.kv_free_pages = 0;             // kv_util = 100%
    snap.mamba_free_slots = 100;        // mamba_util = 0%
    auto plan = agent.Tick(snap);
    EXPECT_FALSE(plan.has_value());
}

TEST(BudgeterTest, ReverseCooldownAllowsSameDirection) {
    SchedulerConfig config = MakeBudgeterConfig();
    config.xpool_reverse_cooldown_s = 1.0;
    BudgetAgent agent(config);

    // Step 1: a kv_to_mamba fire commits.
    agent.OnFireCommitted("kv_to_mamba");

    // Step 2: pressure stays in the SAME direction (mamba still exhausted,
    // kv still free). Cooldown must NOT gate this — same-direction fires
    // can accumulate capacity transfer.
    TinySleep();
    PoolSnapshot snap = MakeBaseSnapshot();
    snap.kv_free_pages = 100;           // kv_util = 0%
    snap.mamba_free_slots = 0;          // mamba_util = 100%
    auto plan = agent.Tick(snap);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->direction, "kv_to_mamba");
}

TEST(BudgeterTest, ReverseCooldownExpiresAfterWindow) {
    SchedulerConfig config = MakeBudgeterConfig();
    // Tiny window so the test stays fast. 20 ms is well above the 2 ms
    // TinySleep used elsewhere and below the timeout budget for unit tests.
    config.xpool_reverse_cooldown_s = 0.02;
    BudgetAgent agent(config);

    agent.OnFireCommitted("kv_to_mamba");

    // Sleep past the cooldown window.
    std::this_thread::sleep_for(std::chrono::milliseconds(30));

    PoolSnapshot snap = MakeBaseSnapshot();
    snap.kv_free_pages = 0;             // kv_util = 100%
    snap.mamba_free_slots = 100;        // mamba_util = 0%
    auto plan = agent.Tick(snap);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->direction, "mamba_to_kv");
}

TEST(BudgeterTest, ReverseCooldownDisabledByZero) {
    SchedulerConfig config = MakeBudgeterConfig();
    config.xpool_reverse_cooldown_s = 0.0;  // disabled
    BudgetAgent agent(config);

    // Same setup as ReverseCooldownSuppressesOppositeDirection but with
    // the cooldown disabled. The opposite-direction plan must fire.
    agent.OnFireCommitted("kv_to_mamba");

    TinySleep();
    PoolSnapshot snap = MakeBaseSnapshot();
    snap.kv_free_pages = 0;
    snap.mamba_free_slots = 100;
    auto plan = agent.Tick(snap);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->direction, "mamba_to_kv");
}

TEST(BudgeterTest, ReverseCooldownNotArmedByCancel) {
    // OnFireCommitted is only called from Scheduler::ApplyXPoolFire (physical
    // commit). The cancel path in Scheduler::CancelXPoolFire goes through
    // ClearPendingFire only, which must NOT arm the cooldown — a cancelled
    // fire is a no-op at the physical layer, so it should not block the
    // budgeter from picking a different direction on the next tick.
    SchedulerConfig config = MakeBudgeterConfig();
    config.xpool_reverse_cooldown_s = 1.0;
    BudgetAgent agent(config);

    // Simulate a kv_to_mamba plan that the actuator then cancels.
    TinySleep();
    PoolSnapshot pressure_snap = MakeBaseSnapshot();
    pressure_snap.kv_free_pages = 100;     // kv_util = 0%
    pressure_snap.mamba_free_slots = 0;    // mamba_util = 100%
    auto fire_plan = agent.Tick(pressure_snap);
    ASSERT_TRUE(fire_plan.has_value());
    EXPECT_EQ(fire_plan->direction, "kv_to_mamba");
    // Cancel path: only ClearPendingFire is called, NOT OnFireCommitted.
    agent.ClearPendingFire();
    EXPECT_EQ(agent.LastFireDirection(), "");  // cooldown not armed

    // Now flip pressure to the opposite direction — must be allowed since
    // the previous fire was cancelled, not committed.
    TinySleep();
    PoolSnapshot snap = MakeBaseSnapshot();
    snap.kv_free_pages = 0;
    snap.mamba_free_slots = 100;
    auto plan = agent.Tick(snap);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->direction, "mamba_to_kv");
}

TEST(BudgeterTest, PendingFireIsLatchedAndClearable) {
    BudgetAgent agent(MakeBudgeterConfig());

    TinySleep();
    PoolSnapshot snap = MakeBaseSnapshot();
    snap.kv_free_pages = 0;                // kv_util = 100%
    snap.mamba_free_slots = 100;           // mamba_util = 0%
    auto plan = agent.Tick(snap);
    ASSERT_TRUE(plan.has_value());

    auto pending = agent.PendingFire();
    ASSERT_TRUE(pending.has_value());
    EXPECT_EQ(pending->op_id, plan->op_id);

    agent.ClearPendingFire();
    EXPECT_FALSE(agent.PendingFire().has_value());
}

// =====================================================================
// S2.3 PressureAdapter tests
// =====================================================================
//
// Weighted queue / retract / paused signals augment the utilisation EWMA
// before the fire-direction comparison and the S2.1 saturation gate.
// Default weights are all 0 so the baseline behaviour is preserved
// (verified implicitly by every test above this section).

TEST(BudgeterTest, PressureAdapterDefaultWeightsAreNoOp) {
    // With all weights at 0, even a large queue/retract burst must not
    // change the fire decision.
    BudgetAgent agent(MakeBudgeterConfig());

    TinySleep();
    PoolSnapshot snap = MakeBaseSnapshot();
    snap.kv_free_pages = 50;        // kv_util = 50%
    snap.mamba_free_slots = 50;     // mamba_util = 50%, delta = 0
    snap.queue_len = 1000;
    snap.retracted_count = 1000;
    snap.paused_count = 1000;
    EXPECT_FALSE(agent.Tick(snap).has_value());
}

TEST(BudgeterTest, PressureAdapterQueueLiftsKvPressure) {
    // With xpool_w_queue > 0, a large queue must lift adjusted KV pressure
    // above mamba so the budgeter fires mamba_to_kv even though raw
    // utilisation is balanced.
    SchedulerConfig config = MakeBudgeterConfig();
    config.xpool_w_queue = 0.5;     // queue full = +0.5 KV pressure
    config.xpool_queue_ref = 10;    // queue saturates at 10 reqs
    BudgetAgent agent(config);

    TinySleep();
    PoolSnapshot snap = MakeBaseSnapshot();
    snap.kv_free_pages = 50;        // kv_util = 50%
    snap.mamba_free_slots = 50;     // mamba_util = 50%
    snap.queue_len = 20;            // > queue_ref => q_norm = 1
    auto plan = agent.Tick(snap);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->direction, "mamba_to_kv");

    // The EWMA accessor reports the smoothed queue depth.
    EXPECT_GT(agent.EwmaQueue(), 0.0);
}

TEST(BudgeterTest, PressureAdapterRetractLiftsKvPressure) {
    SchedulerConfig config = MakeBudgeterConfig();
    config.xpool_w_retract = 0.4;
    config.xpool_retract_ref = 4;
    BudgetAgent agent(config);

    TinySleep();
    PoolSnapshot snap = MakeBaseSnapshot();
    snap.kv_free_pages = 50;            // kv_util = 50%
    snap.mamba_free_slots = 50;         // mamba_util = 50%
    snap.retracted_count = 8;           // r_norm saturates at 1
    auto plan = agent.Tick(snap);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->direction, "mamba_to_kv");
    EXPECT_GT(agent.EwmaRetract(), 0.0);
}

TEST(BudgeterTest, PressureAdapterPausedLiftsMambaPressure) {
    SchedulerConfig config = MakeBudgeterConfig();
    config.xpool_w_paused = 0.5;
    config.xpool_paused_ref = 4;
    // The mamba-headroom guard fires only when n_mamba > headroom; we set
    // bytes_per_page == bytes_per_slot so n_mamba == n_kv == 64 and the
    // headroom default (1000) easily accomodates it.
    config.kv_bytes_per_page = 1024;
    config.mamba_bytes_per_slot = 1024;
    BudgetAgent agent(config);

    TinySleep();
    PoolSnapshot snap = MakeBaseSnapshot();
    snap.kv_free_pages = 50;            // kv_util = 50%
    snap.mamba_free_slots = 50;         // mamba_util = 50%
    snap.paused_count = 8;              // p_norm saturates at 1
    auto plan = agent.Tick(snap);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->direction, "kv_to_mamba");
    EXPECT_GT(agent.EwmaPaused(), 0.0);
}

TEST(BudgeterTest, PressureAdapterCanLiftBelowSaturationGate) {
    // A heavy queue burst must lift adjusted KV pressure across the S2.1
    // saturation gate even when raw utilisation EWMA stays low.
    SchedulerConfig config = MakeBudgeterConfig();
    config.xpool_saturation_low = 0.5;
    config.xpool_w_queue = 0.6;
    config.xpool_queue_ref = 10;
    BudgetAgent agent(config);

    TinySleep();
    PoolSnapshot snap = MakeBaseSnapshot();
    snap.kv_free_pages = 70;        // kv_util = 30% (below gate)
    snap.mamba_free_slots = 100;    // mamba_util = 0% (below gate)
    snap.queue_len = 20;            // adj_kv = 0.30 + 0.6*1.0 = 0.90 >> 0.5
    auto plan = agent.Tick(snap);
    ASSERT_TRUE(plan.has_value())
        << "queue burst must lift adj_kv above xpool_saturation_low";
    EXPECT_EQ(plan->direction, "mamba_to_kv");
}

TEST(BudgeterTest, PressureAdapterRespectsReverseCooldown) {
    // Even with a strong queue signal, an opposite-direction fire must
    // still be suppressed during the reverse cooldown window.  This
    // catches accidental interactions between S2.2 and S2.3 — they are
    // independent gates that compose conservatively.
    SchedulerConfig config = MakeBudgeterConfig();
    config.xpool_reverse_cooldown_s = 1.0;
    config.xpool_w_queue = 0.5;
    config.xpool_queue_ref = 10;
    BudgetAgent agent(config);

    agent.OnFireCommitted("kv_to_mamba");
    TinySleep();
    PoolSnapshot snap = MakeBaseSnapshot();
    snap.kv_free_pages = 50;
    snap.mamba_free_slots = 50;
    snap.queue_len = 20;
    EXPECT_FALSE(agent.Tick(snap).has_value());
}

// S2.3 w_paused / S2.6 migration -----------------------------------------------

// DeferredCount() must increment for each kDefer decision and reset after Tick.
TEST(BudgeterTest, DeferredCountAccumulatesAndResetsOnTick) {
    SchedulerConfig config = MakeBudgeterConfig();
    config.enable_admitter = true;
    config.xpool_saturation_low = 0.0;  // no gate
    BudgetAgent agent(config);

    // Starved snapshot: both pools empty, no active KV → admitter defers.
    PoolSnapshot starved;
    starved.kv_free_pages = 0;
    starved.mamba_free_slots = 0;
    starved.kv_total_pages = 100;
    starved.mamba_total_slots = 100;
    starved.kv_active_pages = 0;

    EXPECT_EQ(agent.DeferredCount(), 0);
    agent.OnRequestArrival(128, starved);
    EXPECT_EQ(agent.DeferredCount(), 1);
    agent.OnRequestArrival(64, starved);
    EXPECT_EQ(agent.DeferredCount(), 2);

    // After Tick() the counter resets.
    TinySleep();
    agent.Tick(starved);
    EXPECT_EQ(agent.DeferredCount(), 0);
}

// When the admitter decides kCrossMigrate, a migration plan is latched and
// DeferredCount stays 0 (migrate != defer).
TEST(BudgeterTest, MigratePlanLatchedOnCrossMigrate) {
    SchedulerConfig config = MakeBudgeterConfig();
    config.enable_admitter = true;
    config.xpool_saturation_low = 0.0;
    BudgetAgent agent(config);

    PoolSnapshot snap;
    snap.kv_free_pages = 0;
    snap.mamba_free_slots = 0;
    snap.kv_total_pages = 100;
    snap.mamba_total_slots = 100;
    snap.kv_active_pages = 512;  // victim available

    EXPECT_FALSE(agent.PendingMigrate().has_value());
    agent.OnRequestArrival(64, snap);

    ASSERT_TRUE(agent.PendingMigrate().has_value());
    EXPECT_EQ(agent.PendingMigrate()->pages_needed, 64);
    // kCrossMigrate must NOT increment the defer counter.
    EXPECT_EQ(agent.DeferredCount(), 0);
}

// CommittedMigrateCount increments via IncrCommittedMigrate and ClearPendingMigrate
// clears the latch.
TEST(BudgeterTest, CommitAndClearMigratePlan) {
    SchedulerConfig config = MakeBudgeterConfig();
    config.enable_admitter = true;
    config.xpool_saturation_low = 0.0;
    BudgetAgent agent(config);

    PoolSnapshot snap;
    snap.kv_free_pages = 0;
    snap.mamba_free_slots = 0;
    snap.kv_total_pages = 100;
    snap.mamba_total_slots = 100;
    snap.kv_active_pages = 256;

    agent.OnRequestArrival(128, snap);
    ASSERT_TRUE(agent.PendingMigrate().has_value());
    EXPECT_EQ(agent.CommittedMigrateCount(), 0);

    agent.IncrCommittedMigrate();
    EXPECT_EQ(agent.CommittedMigrateCount(), 1);

    agent.ClearPendingMigrate();
    EXPECT_FALSE(agent.PendingMigrate().has_value());
}

// paused_count fed into PoolSnapshot should be reflected in EwmaPaused after Tick.
TEST(BudgeterTest, PausedCountFlowsThroughEwmaViaDeferredCounter) {
    SchedulerConfig config = MakeBudgeterConfig();
    config.enable_admitter = true;
    config.xpool_saturation_low = 0.0;
    BudgetAgent agent(config);

    // Accumulate two defer decisions so DeferredCount() == 2.
    PoolSnapshot starved;
    starved.kv_free_pages = 0;
    starved.mamba_free_slots = 0;
    starved.kv_total_pages = 100;
    starved.mamba_total_slots = 100;
    starved.kv_active_pages = 0;

    agent.OnRequestArrival(128, starved);
    agent.OnRequestArrival(64, starved);
    EXPECT_EQ(agent.DeferredCount(), 2);

    // Build a snapshot with paused_count == DeferredCount (as MakePoolSnapshot
    // would do) and tick: EwmaPaused must be positive afterward.
    PoolSnapshot snap_with_paused = starved;
    snap_with_paused.paused_count = agent.DeferredCount();
    TinySleep();
    agent.Tick(snap_with_paused);
    EXPECT_GT(agent.EwmaPaused(), 0.0);
}

}  // namespace tokenspeed::test
