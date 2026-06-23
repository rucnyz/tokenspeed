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

#include "budgeter/admitter.h"
#include "budgeter/cost_model.h"

namespace tokenspeed::test {

TEST(AdmitterSevenTest, OwnFreeWhenBothPoolsHaveCapacity) {
    CostModel model;
    Admitter admitter(model);
    PoolSnapshot snap{.kv_free_pages = 128, .mamba_free_slots = 8, .queue_len = 0};
    auto decision = admitter.DecideForRequest(32, snap);
    EXPECT_EQ(decision.action, AdmitAction::kOwnFree);
}

TEST(AdmitterSevenTest, DeferWhenBothPoolsStarved) {
    CostModel model;
    Admitter admitter(model);
    PoolSnapshot snap{.kv_free_pages = 0, .mamba_free_slots = 0, .queue_len = 3};
    auto decision = admitter.DecideForRequest(128, snap);
    EXPECT_EQ(decision.action, AdmitAction::kDefer);
}

// S2.6 tests -------------------------------------------------------------------

// When both pools are starved but there are enough active KV pages to retract
// a victim, the admitter should return kCrossMigrate instead of kDefer.
TEST(AdmitterSevenTest, CrossMigrateWhenBothStarvedButActiveKvAvailable) {
    CostModel model;
    Admitter admitter(model);
    PoolSnapshot snap{
        .kv_free_pages = 0,
        .mamba_free_slots = 0,
        .queue_len = 2,
        .kv_active_pages = 256,  // enough to satisfy the 64-page request
    };
    auto decision = admitter.DecideForRequest(64, snap);
    EXPECT_EQ(decision.action, AdmitAction::kCrossMigrate);
    EXPECT_EQ(decision.pages_needed, 64);
    EXPECT_GT(decision.cost_us, 0.0);
}

// When both pools are starved AND kv_active_pages is insufficient, kDefer
// still takes priority (no victim large enough to retract).
TEST(AdmitterSevenTest, DeferWhenBothStarvedAndActiveKvInsufficient) {
    CostModel model;
    Admitter admitter(model);
    PoolSnapshot snap{
        .kv_free_pages = 0,
        .mamba_free_slots = 0,
        .queue_len = 1,
        .kv_active_pages = 16,  // less than the 64-page request needs
    };
    auto decision = admitter.DecideForRequest(64, snap);
    EXPECT_EQ(decision.action, AdmitAction::kDefer);
}

// kCrossMigrate cost must exceed zero (it models write-back + eviction).
TEST(AdmitterSevenTest, CrossMigrateCostIsPositive) {
    CostModel model;
    // Use a non-trivial c_m so CMigrateUs() != 0.
    model.eviction_config.c_m = 500.0;
    Admitter admitter(model);
    PoolSnapshot snap{
        .kv_free_pages = 0,
        .mamba_free_slots = 0,
        .queue_len = 0,
        .kv_active_pages = 512,
    };
    auto decision = admitter.DecideForRequest(128, snap);
    EXPECT_EQ(decision.action, AdmitAction::kCrossMigrate);
    EXPECT_GT(decision.cost_us, 0.0);
}

}  // namespace tokenspeed::test
