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

TEST(BudgeterTest, PendingFireIsLatchedAndClearable) {
    BudgetAgent agent(MakeBudgeterConfig());

    TinySleep();
    PoolSnapshot snap = MakeBaseSnapshot();
    snap.kv_free_pages = 0;                // kv_util = 100%
    snap.mamba_free_slots = 100;           // mamba_util = 0%
    auto plan = agent.Tick(snap);
    ASSERT_TRUE(plan.has_value());

    // The plan stays latched until explicitly cleared.
    auto pending = agent.PendingFire();
    ASSERT_TRUE(pending.has_value());
    EXPECT_EQ(pending->op_id, plan->op_id);

    agent.ClearPendingFire();
    EXPECT_FALSE(agent.PendingFire().has_value());
}

}  // namespace tokenspeed::test
