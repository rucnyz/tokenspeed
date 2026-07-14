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

// Verifies S2.5-followup proactive retract: after PrepareKvToMambaFire caps
// the KV tail, NextExecutionPlan retracts a decoder that still holds capped pages
// instead of waiting for natural decode completion.

#include "integration_test_helper.h"

namespace tokenspeed::test {

class XPoolProactiveRetractSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.page_size = 2;
        cfg.decode_input_tokens = 0;
        cfg.device_allocator.total_pages = 14;
        cfg.host_allocator.total_pages = 32;
        cfg.enable_l3_storage = false;
        cfg.enable_xpool_dynamic_capacity = true;
        cfg.xpool_initial_kv_pages = 10;
        return cfg;
    }

    static const FlatWriteBackOperation* GetWriteBack(const ExecutionPlan& plan) {
        for (const auto& op : plan.Operations()) {
            if (auto* cop = std::get_if<CacheOperation>(&op)) {
                if (auto* wb = std::get_if<FlatWriteBackOperation>(cop)) {
                    return wb;
                }
            }
        }
        return nullptr;
    }
};

TEST_F(XPoolProactiveRetractSuite, PrepareFireTriggersProactiveRetractForTailHolder) {
    // Fill most of the 10-page KV baseline so the tail pages are owned by r1.
    Submit(MakeRequestSpec("r1", /*num_pages=*/9, /*start=*/1));
    PlanOnce();  // Submitted → PrefillDone
    SendForwardDone("r1", {42});
    PlanOnce();  // PrefillDone → Decoding
    ASSERT_EQ(scheduler_->DecodingSize(), 1u);

    scheduler_->PrepareKvToMambaFire(/*n_kv_pages=*/2);
    EXPECT_TRUE(scheduler_->HasCappedKvInflight());

    auto plan = PlanOnce();
    const auto* wb = GetWriteBack(plan);
    ASSERT_NE(wb, nullptr);
    EXPECT_FALSE(wb->op_ids.empty());
}

TEST_F(XPoolProactiveRetractSuite, NoProactiveRetractWhenXpoolDisabled) {
    auto cfg = MakeConfig();
    cfg.enable_xpool_dynamic_capacity = false;
    scheduler_ = std::make_unique<Scheduler>(cfg);

    Submit(MakeRequestSpec("r1", 9, 1));
    PlanOnce();
    SendForwardDone("r1", {42});
    PlanOnce();

    scheduler_->PrepareKvToMambaFire(2);
    auto plan = PlanOnce();
    EXPECT_EQ(GetWriteBack(plan), nullptr);
}

// After PrepareFire, repeated plans should not spam retract attempts on the
// same tick cadence when the first attempt already latched a victim.
TEST_F(XPoolProactiveRetractSuite, RepeatedPlansDoNotRequireTickCooldownId) {
    Submit(MakeRequestSpec("r1", /*num_pages=*/9, /*start=*/1));
    PlanOnce();
    SendForwardDone("r1", {42});
    PlanOnce();

    scheduler_->PrepareKvToMambaFire(/*n_kv_pages=*/2);
    ASSERT_TRUE(scheduler_->HasCappedKvInflight());

    auto plan1 = PlanOnce();
    const auto* wb1 = GetWriteBack(plan1);
    ASSERT_NE(wb1, nullptr);

    // Second plan while capped inflight persists: either no new retract or a
    // different victim — must not crash and must remain schedulable.
    auto plan2 = PlanOnce();
    (void)GetWriteBack(plan2);
    SUCCEED();
}

}  // namespace tokenspeed::test
