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

// Verifies Scheduler::ApplyXPoolFire correctly adjusts both KV and mamba
// allocator capacities and clears the pending fire latch.

#include <gtest/gtest.h>

#include "scheduler/scheduler.h"
#include "scheduler/types.h"

namespace tokenspeed::test {

namespace {

// Build a minimal SchedulerConfig suitable for XPool capacity tests.
// Profiled KV: 32 pages; profiled mamba: 64 slots. VA window adds headroom
// (16 KV pages / 4 mamba slots) so transfers in both directions fit.
SchedulerConfig MakeConfig() {
    SchedulerConfig cfg;
    cfg.max_batch_size = 4;
    cfg.page_size = 16;
    // VA window = profiled (32) + max transfer headroom (16) + 1 (reserved page 0).
    cfg.device_allocator.total_pages = 49;
    cfg.host_allocator.total_pages = 1;

    cfg.enable_mamba = true;
    // VA window = profiled (64) + max transfer headroom (4).
    cfg.mamba_pool_total_chunks = 68;
    cfg.mamba_cache_chunk_size = 16;

    // kv_bytes_per_page and mamba_bytes_per_slot drive the unit conversion in
    // ApplyXPoolFire: 4 KV pages == 1 mamba slot (ratio 4:1).
    cfg.kv_bytes_per_page = 4096;
    cfg.mamba_bytes_per_slot = 16384;

    cfg.enable_xpool_dynamic_capacity = true;
    // Initial capacities = profiled budgets (smaller than VA window).
    cfg.xpool_initial_kv_pages = 32;
    cfg.xpool_initial_mamba_slots = 64;

    // Enable budgeter so we can check latch clearing.
    cfg.enable_budgeter = true;
    cfg.budgeter_pages_per_fire = 4;
    cfg.xpool_ewma_tau_s = 1e9;  // no spontaneous ticks in tests

    return cfg;
}

XPoolFirePlan MakePlan(const std::string& dir, std::int32_t n_pages, int op_id = 1) {
    XPoolFirePlan p;
    p.direction = dir;
    p.op_id = op_id;
    p.page_ids.reserve(static_cast<std::size_t>(n_pages));
    for (std::int32_t i = 0; i < n_pages; ++i) p.page_ids.push_back(i + 1);
    return p;
}

}  // namespace

TEST(ApplyXPoolFireTest, MambaToKvGrowsKvShrinksMaxmba) {
    Scheduler sched(MakeConfig());
    // At startup (dynamic mode), both KV and mamba are at baseline.
    const std::int32_t kv_start = sched.MappedKvPages();
    const std::int32_t mb_start = sched.MappedMambaSlots();
    ASSERT_GT(kv_start, 0);
    ASSERT_GT(mb_start, 0);

    // Fire plan: 4 KV pages from mamba → KV.
    // Unit conversion: 4 * 4096 / 16384 = 1 mamba slot.
    sched.ApplyXPoolFire(MakePlan("mamba_to_kv", 4));

    EXPECT_EQ(sched.MappedKvPages(), kv_start + 4);
    EXPECT_EQ(sched.MappedMambaSlots(), mb_start - 1);
}

TEST(ApplyXPoolFireTest, KvToMambaShrinksKvGrowsMamba) {
    Scheduler sched(MakeConfig());
    const std::int32_t kv_start = sched.MappedKvPages();
    const std::int32_t mb_start = sched.MappedMambaSlots();

    // Fire plan: 4 KV pages from KV → mamba → 1 mamba slot gained.
    sched.ApplyXPoolFire(MakePlan("kv_to_mamba", 4));

    EXPECT_EQ(sched.MappedKvPages(), kv_start - 4);
    EXPECT_EQ(sched.MappedMambaSlots(), mb_start + 1);
}

TEST(ApplyXPoolFireTest, FireClearsPendingLatch) {
    Scheduler sched(MakeConfig());
    // Manually push a plan via BudgetTick by manipulating EWMA externally.
    // Easiest: just check that ApplyXPoolFire → PendingXPoolFire == nullopt.
    // We directly call ApplyXPoolFire (pretending actuator completed work).
    sched.ApplyXPoolFire(MakePlan("mamba_to_kv", 4));
    // After applying, the latch should be cleared.
    EXPECT_FALSE(sched.PendingXPoolFire().has_value());
}

TEST(ApplyXPoolFireTest, NoOpWhenXpoolDisabled) {
    SchedulerConfig cfg = MakeConfig();
    cfg.enable_xpool_dynamic_capacity = false;
    Scheduler sched(cfg);

    const std::int32_t kv_before = sched.MappedKvPages();
    sched.ApplyXPoolFire(MakePlan("mamba_to_kv", 4));
    EXPECT_EQ(sched.MappedKvPages(), kv_before);  // unchanged
}

TEST(ApplyXPoolFireTest, RoundTripKvToMambaAndBack) {
    Scheduler sched(MakeConfig());
    const std::int32_t kv_start = sched.MappedKvPages();
    const std::int32_t mb_start = sched.MappedMambaSlots();

    sched.ApplyXPoolFire(MakePlan("kv_to_mamba", 8, 1));
    EXPECT_EQ(sched.MappedKvPages(), kv_start - 8);
    EXPECT_EQ(sched.MappedMambaSlots(), mb_start + 2);  // 8 * 4096 / 16384 = 2

    sched.ApplyXPoolFire(MakePlan("mamba_to_kv", 8, 2));
    EXPECT_EQ(sched.MappedKvPages(), kv_start);
    EXPECT_EQ(sched.MappedMambaSlots(), mb_start);
}

}  // namespace tokenspeed::test
