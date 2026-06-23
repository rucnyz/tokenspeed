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

// Verifies Scheduler::SetMaxBatchSize / S2.7 dynamic admission cap.
//
// Tests cover the explicit setter (with clamping), and the BudgetTick
// auto-sync path: when enable_dynamic_admission_cap is true, each tick
// must clamp max_batch_size to mamba_mapped_slots.  When the feature is
// off, the cap is untouched (regression guard for unrelated workloads).

#include <gtest/gtest.h>

#include "scheduler/scheduler.h"
#include "scheduler/types.h"

namespace tokenspeed::test {

namespace {

SchedulerConfig MakeConfig(bool enable_dyn_cap) {
    SchedulerConfig cfg;
    cfg.max_batch_size = 64;
    cfg.page_size = 16;
    cfg.device_allocator.total_pages = 256;
    cfg.host_allocator.total_pages = 1;

    cfg.enable_mamba = true;
    cfg.mamba_pool_total_chunks = 128;
    cfg.mamba_cache_chunk_size = 16;
    cfg.kv_bytes_per_page = 4096;
    cfg.mamba_bytes_per_slot = 16384;

    cfg.enable_xpool_dynamic_capacity = true;
    cfg.xpool_initial_kv_pages = 128;
    cfg.xpool_initial_mamba_slots = 32;  // chosen < max_batch_size to expose the clamp

    cfg.enable_budgeter = true;
    cfg.budgeter_pages_per_fire = 4;
    // Long tau so BudgetTick doesn't fire spontaneously inside this test;
    // we only care about the admission-cap sync side-effect.
    cfg.xpool_ewma_tau_s = 1e9;
    // Disable saturation gate so EWMA still updates but no fire is issued.
    cfg.xpool_saturation_low = 0.0;
    // Margin big enough that the pressure delta from initial capacity
    // (kv 0%, mamba 0%) never crosses it.
    cfg.xpool_nb_margin = 0.99;

    cfg.enable_dynamic_admission_cap = enable_dyn_cap;
    return cfg;
}

}  // namespace

TEST(DynamicAdmissionCapTest, SetMaxBatchSizeClampsUpper) {
    // Setter must clamp above the boot-time max_batch_size: the
    // ReqPoolAllocator cannot grow, so raising the cap above the
    // original is forbidden.
    Scheduler sched(MakeConfig(/*enable_dyn_cap=*/false));
    EXPECT_EQ(sched.MaxBatchSize(), 64);
    sched.SetMaxBatchSize(1000);
    EXPECT_EQ(sched.MaxBatchSize(), 64)
        << "raising the cap above the boot-time max_batch_size must be clamped";
}

TEST(DynamicAdmissionCapTest, SetMaxBatchSizeClampsLower) {
    // Setter must always leave a forward-progress floor of 1; a value
    // of zero or negative collapses to 1 so the scheduler can still
    // drain in-flight requests.
    Scheduler sched(MakeConfig(/*enable_dyn_cap=*/false));
    sched.SetMaxBatchSize(0);
    EXPECT_EQ(sched.MaxBatchSize(), 1);
    sched.SetMaxBatchSize(-100);
    EXPECT_EQ(sched.MaxBatchSize(), 1);
}

TEST(DynamicAdmissionCapTest, SetMaxBatchSizeMidRange) {
    Scheduler sched(MakeConfig(/*enable_dyn_cap=*/false));
    sched.SetMaxBatchSize(16);
    EXPECT_EQ(sched.MaxBatchSize(), 16);
    sched.SetMaxBatchSize(48);
    EXPECT_EQ(sched.MaxBatchSize(), 48);
}

TEST(DynamicAdmissionCapTest, BudgetTickSyncsCapWhenEnabled) {
    // enable_dynamic_admission_cap=true: BudgetTick must clamp
    // max_batch_size to MappedMambaSlots, which is 32 here.
    Scheduler sched(MakeConfig(/*enable_dyn_cap=*/true));
    ASSERT_EQ(sched.MaxBatchSize(), 64);
    ASSERT_EQ(sched.MappedMambaSlots(), 32);

    sched.BudgetTick();
    EXPECT_EQ(sched.MaxBatchSize(), 32)
        << "dynamic admission cap must clamp to mamba_mapped_slots";
}

TEST(DynamicAdmissionCapTest, BudgetTickLeavesCapAloneWhenDisabled) {
    // Regression guard: with the feature off, the cap stays at its
    // boot-time value across budget ticks regardless of mamba size.
    Scheduler sched(MakeConfig(/*enable_dyn_cap=*/false));
    ASSERT_EQ(sched.MaxBatchSize(), 64);
    sched.BudgetTick();
    EXPECT_EQ(sched.MaxBatchSize(), 64);
}

}  // namespace tokenspeed::test
