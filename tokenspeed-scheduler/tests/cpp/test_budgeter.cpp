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

TEST(BudgeterTest, KvPressureFiresMambaToKv) {
    BudgetAgent agent(MakeBudgeterConfig());

    TinySleep();
    // KV exhausted (util=1), mamba has plenty of free slots (util=0).
    PoolSnapshot snap{.kv_free_pages = 0, .mamba_free_slots = 100};
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
    PoolSnapshot snap{.kv_free_pages = 100, .mamba_free_slots = 0};
    auto plan = agent.Tick(snap);

    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->direction, "kv_to_mamba");
    EXPECT_EQ(plan->page_ids.size(), 64u);
}

TEST(BudgeterTest, BalancedPoolsDoNotFire) {
    BudgetAgent agent(MakeBudgeterConfig());

    TinySleep();
    // Both pools have capacity -> equal pressure -> no transfer.
    PoolSnapshot snap{.kv_free_pages = 50, .mamba_free_slots = 50};
    EXPECT_FALSE(agent.Tick(snap).has_value());
}

TEST(BudgeterTest, MambaFloorBlocksMambaToKvFire) {
    SchedulerConfig config = MakeBudgeterConfig();
    config.xpool_mamba_floor_slots = 50;  // keep at least 50 mamba slots
    BudgetAgent agent(config);

    TinySleep();
    // KV exhausted and mamba has free slots, but below the protective floor:
    // the budgeter must not steal mamba capacity.
    PoolSnapshot snap{.kv_free_pages = 0, .mamba_free_slots = 10};
    EXPECT_FALSE(agent.Tick(snap).has_value());
}

TEST(BudgeterTest, PendingFireIsLatchedAndClearable) {
    BudgetAgent agent(MakeBudgeterConfig());

    TinySleep();
    PoolSnapshot snap{.kv_free_pages = 0, .mamba_free_slots = 100};
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
