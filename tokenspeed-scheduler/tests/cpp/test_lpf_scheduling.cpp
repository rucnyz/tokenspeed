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

#include "integration_test_helper.h"

namespace tokenspeed::test {

// ============================================================
//  LPF (Longest-Prefix-First) Scheduling
// ============================================================

class LPFSchedulingTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.enable_l3_storage = false;
        return cfg;
    }

    static const FlatForwardOperation* GetForwardOp(const ExecutionPlan& plan) {
        for (const auto& op : plan.Operations()) {
            if (auto* f = std::get_if<FlatForwardOperation>(&op)) return f;
        }
        return nullptr;
    }

    void PopulatePrefixCache() {
        // Submit r0 with tokens [1,2,3,4] (2 pages at page_size=2), run to completion.
        Submit(MakeRequestSpec("r0", 2, 1));
        PlanOnce();  // r0 prefills
        SendForwardDone("r0", {42});
        PlanOnce();  // r0 -> Decoding
        SendFinish("r0");
        PlanOnce();  // r0 -> Finished, pages inserted into prefix cache
    }
};

// z_hot shares prefix with cache (match_depth=2), a_cold does not (match_depth=0).
// Without LPF: a_cold < z_hot alphabetically -> a_cold scheduled first.
// With LPF: z_hot has longer prefix match -> scheduled first despite higher ID.
TEST_F(LPFSchedulingTestSuite, LPF_PrefixMatchOrderedFirst) {
    PopulatePrefixCache();

    // z_hot: tokens [1,2,3,4,5,6] -- first 2 pages match cache
    RequestSpec hot_spec{
        .request_id = "z_hot",
        .tokens = {1, 2, 3, 4, 5, 6},
    };
    // a_cold: tokens [100,101,102,103] -- no prefix match
    RequestSpec cold_spec{
        .request_id = "a_cold",
        .tokens = {100, 101, 102, 103},
    };

    Submit({cold_spec, hot_spec});

    auto plan = PlanOnce();
    const auto* fwd = GetForwardOp(plan);
    ASSERT_NE(fwd, nullptr);
    ASSERT_GE(fwd->request_ids.size(), 2u);

    int hot_pos = -1, cold_pos = -1;
    for (std::size_t i = 0; i < fwd->request_ids.size(); ++i) {
        if (fwd->request_ids[i] == "z_hot") hot_pos = static_cast<int>(i);
        if (fwd->request_ids[i] == "a_cold") cold_pos = static_cast<int>(i);
    }
    ASSERT_NE(hot_pos, -1) << "z_hot (long prefix) must be scheduled";
    ASSERT_NE(cold_pos, -1) << "a_cold must also be scheduled";
    EXPECT_LT(hot_pos, cold_pos) << "LPF: z_hot (prefix match=2) before a_cold (prefix match=0)";
}

// When both requests have same prefix match depth, fall back to ID ordering.
TEST_F(LPFSchedulingTestSuite, LPF_SameDepthFallsBackToIdOrder) {
    // No prefix cache populated -- both requests have match_depth=0.
    RequestSpec ra_spec{
        .request_id = "ra",
        .tokens = {10, 11, 12, 13},
    };
    RequestSpec rb_spec{
        .request_id = "rb",
        .tokens = {20, 21, 22, 23},
    };

    Submit({rb_spec, ra_spec});

    auto plan = PlanOnce();
    const auto* fwd = GetForwardOp(plan);
    ASSERT_NE(fwd, nullptr);
    ASSERT_GE(fwd->request_ids.size(), 2u);

    int ra_pos = -1, rb_pos = -1;
    for (std::size_t i = 0; i < fwd->request_ids.size(); ++i) {
        if (fwd->request_ids[i] == "ra") ra_pos = static_cast<int>(i);
        if (fwd->request_ids[i] == "rb") rb_pos = static_cast<int>(i);
    }
    ASSERT_NE(ra_pos, -1);
    ASSERT_NE(rb_pos, -1);
    // Same match depth (0) -> tie-break by ID: "ra" < "rb"
    EXPECT_LT(ra_pos, rb_pos) << "Same prefix depth: tie-break by request ID";
}

}  // namespace tokenspeed::test
