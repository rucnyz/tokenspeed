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

#include "budgeter/admitter.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace tokenspeed {

namespace {

constexpr AdmitAction kTieBreakOrder[] = {
    AdmitAction::kOwnFree,      AdmitAction::kCrossFree,   AdmitAction::kOwnEvict,
    AdmitAction::kCrossEvict,   AdmitAction::kOwnMigrate,  AdmitAction::kCrossMigrate,
    AdmitAction::kDefer,
};

int TieRank(AdmitAction action) {
    for (std::size_t i = 0; i < sizeof(kTieBreakOrder) / sizeof(kTieBreakOrder[0]); ++i) {
        if (kTieBreakOrder[i] == action) {
            return static_cast<int>(i);
        }
    }
    return static_cast<int>(sizeof(kTieBreakOrder) / sizeof(kTieBreakOrder[0]));
}

}  // namespace

Admitter::Admitter(CostModel cost_model) : cost_model_{std::move(cost_model)} {}

AdmitDecision Admitter::decide(std::int32_t x_tokens, std::int32_t x_eff, const PoolSnapshot& snapshot, bool cross,
                               const std::string& direction) const {
    AdmitDecision best;
    best.action = AdmitAction::kDefer;
    best.cost_us = cost_model_.WQueueUs() * snapshot.queue_len;
    best.direction = direction;

    auto consider = [&](AdmitAction action, double cost, bool feasible) {
        if (!feasible) {
            return;
        }
        if (cost < best.cost_us || (cost == best.cost_us && TieRank(action) < TieRank(best.action))) {
            best.action = action;
            best.cost_us = cost;
            best.pages_needed = x_eff;
        }
    };

    if (!cross) {
        consider(AdmitAction::kOwnFree, 0.0, snapshot.kv_free_pages >= x_tokens);
        consider(AdmitAction::kOwnEvict, cost_model_.CEvictUs(x_tokens, 1),
                 snapshot.kv_free_pages + snapshot.kv_evictable_pages >= x_tokens);
        return best;
    }

    const double xfer = cost_model_.CXferUs(x_eff);
    consider(AdmitAction::kCrossFree, xfer, snapshot.mamba_free_slots >= x_eff);
    consider(AdmitAction::kCrossEvict, xfer + cost_model_.CEvictUs(x_eff, 1),
             snapshot.mamba_free_slots + snapshot.mamba_evictable_slots >= x_eff);
    return best;
}

AdmitDecision Admitter::DecideForRequest(std::int32_t prompt_tokens, const PoolSnapshot& snapshot,
                                         std::int32_t mamba_need_slots) const {
    const std::int32_t x_eff = static_cast<std::int32_t>(
        std::ceil(static_cast<double>(prompt_tokens) / static_cast<double>(pages_per_fire_)) * pages_per_fire_);

    const bool kv_enough = snapshot.kv_free_pages >= prompt_tokens;
    const bool mamba_enough = snapshot.mamba_free_slots >= mamba_need_slots;

    if (kv_enough && mamba_enough) {
        return decide(prompt_tokens, x_eff, snapshot, false, "");
    }
    if (!kv_enough && mamba_enough) {
        return decide(prompt_tokens, x_eff, snapshot, true, "mamba_to_kv");
    }
    if (kv_enough && !mamba_enough) {
        return decide(mamba_need_slots, mamba_need_slots, snapshot, true, "kv_to_mamba");
    }
    AdmitDecision defer;
    defer.action = AdmitAction::kDefer;
    defer.cost_us = cost_model_.WQueueUs() * snapshot.queue_len;
    return defer;
}

}  // namespace tokenspeed
