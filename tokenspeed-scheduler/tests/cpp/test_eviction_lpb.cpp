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
#include <memory>
#include <thread>

#include "unit_test_helper.h"
#include "resource/allocator/page_allocator.h"
#include "resource/eviction_config.h"
#include "resource/kv_prefix_cache/eviction.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"
#include "resource/radix_tree/tree_node.h"
#include "resource/types.h"

namespace tokenspeed::test {

class EvictionLPBTest : public ::testing::Test {
protected:
    static constexpr int32_t kPageSize = 4;
    static constexpr int32_t kTotalPages = 64;
    static constexpr int64_t kBytesPerPage = 4096;

    EvictionConfig MakeLPBConfig() const {
        EvictionConfig cfg;
        cfg.policy = EvictionPolicy::kLpb;
        cfg.lpb_window_s = 60.0;
        cfg.lpb_hit_deque_maxlen = 4096;
        cfg.kv_bytes_per_page = kBytesPerPage;
        return cfg;
    }

    void SetUp() override {
        device_alloc_ = std::make_unique<PageAllocator>(kPageSize, kTotalPages);
        cache_ = std::make_unique<KVPrefixCache>(device_alloc_.get(), /*host=*/nullptr,
                                                   /*enable_l3_storage=*/false,
                                                   /*disable_prefix_cache=*/false, MakeLPBConfig());
    }

    InsertResult Insert(int32_t num_pages, token_t start = 1, TreeNode* from = nullptr) {
        auto tokens = MakeAlignedTokens(num_pages, kPageSize, start);
        auto pages = device_alloc_->Allocate(num_pages);
        return cache_->Insert<ResourceType::Device>(tokens, /*prefix_pages=*/{}, std::move(pages), {}, from);
    }

    std::unique_ptr<PageAllocator> device_alloc_;
    std::unique_ptr<KVPrefixCache> cache_;
};

TEST_F(EvictionLPBTest, LowerHitCountEvictedFirst) {
    Insert(1, /*start=*/1);
    Insert(1, /*start=*/5);

    auto match_hot = cache_->Match(MakeAlignedTokens(1, kPageSize, 5));
    for (int i = 0; i < 5; ++i) {
        cache_->Match(MakeAlignedTokens(1, kPageSize, 5));
    }

    token_t first_evicted_start = -1;
    cache_->GetDeviceManager().SetEvictionCallback([&](TreeNode* n) {
        if (first_evicted_start == -1 && !n->Tokens().empty()) {
            first_evicted_start = n->Tokens().front();
        }
    });

    const int32_t initial = device_alloc_->AvailablePages();
    bool ok = cache_->EnsureCapacityByEvict<ResourceType::Device>(initial + 1);
    EXPECT_TRUE(ok);
    EXPECT_EQ(first_evicted_start, 1);
}

TEST_F(EvictionLPBTest, MatchRefreshesStaleLpbPriority) {
    // Regression test for the HiMA Phase 3 (S2.4) stale-priority bug:
    // RecordHit() during a Match() walk increments hit_times_ on a node,
    // but the eviction priority cached in the sorted leaves set is only
    // recomputed by UpdateLeaves(). Without a refresh, every node looks
    // like it has 0 hits to the eviction loop, so LPB silently degenerates
    // to "evict oldest" (i.e. LRU as a tiebreaker on priority=0). This
    // test pins fresh-priority behavior: an old-but-hot node must outlive
    // a newer-but-cold one even though stale-LPB would have killed the
    // old node first.
    Insert(1, /*start=*/1);                             // node A, ts_A
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    Insert(1, /*start=*/5);                             // node B, ts_B (newer)
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    Insert(1, /*start=*/9);                             // node C, ts_C (newest)

    // Match node A several times so its hit count is well above 0.
    for (int i = 0; i < 4; ++i) {
        cache_->Match(MakeAlignedTokens(1, kPageSize, 1));
    }
    // B and C are never matched, so their hit counts stay at 0.

    token_t first_evicted = -1;
    cache_->GetDeviceManager().SetEvictionCallback([&](TreeNode* n) {
        if (first_evicted == -1 && !n->Tokens().empty()) {
            first_evicted = n->Tokens().front();
        }
    });

    const int32_t initial = device_alloc_->AvailablePages();
    bool ok = cache_->EnsureCapacityByEvict<ResourceType::Device>(initial + 1);
    EXPECT_TRUE(ok);

    // With fresh-priority LPB, A has priority > 0 and B/C have priority 0;
    // among {B, C} the older B (smaller ts) sorts first and must be evicted.
    // Stale-priority LPB would have evicted A here (oldest priority-0 node).
    EXPECT_EQ(first_evicted, 5)
        << "LPB must evict cold-but-newer B before hot-but-older A; "
        << "got first_evicted=" << first_evicted
        << " (== 1 means stale priority regressed the fix; == 9 means "
        << "tie-break order differs from insert order)";
}

TEST_F(EvictionLPBTest, LRUModeEquivalentToBaseline) {
    EvictionConfig lru_cfg;
    lru_cfg.policy = EvictionPolicy::kLru;
    lru_cfg.kv_bytes_per_page = kBytesPerPage;
    KVPrefixCache lru_cache(device_alloc_.get(), nullptr,
                            /*enable_l3_storage=*/false, /*disable_prefix_cache=*/false, lru_cfg);

    auto tokens_a = MakeAlignedTokens(1, kPageSize, 1);
    auto tokens_b = MakeAlignedTokens(1, kPageSize, 5);
    lru_cache.Insert<ResourceType::Device>(tokens_a, {}, device_alloc_->Allocate(1), {}, nullptr);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    lru_cache.Insert<ResourceType::Device>(tokens_b, {}, device_alloc_->Allocate(1), {}, nullptr);

    token_t first_evicted = -1;
    lru_cache.GetDeviceManager().SetEvictionCallback([&](TreeNode* n) {
        if (first_evicted == -1 && !n->Tokens().empty()) {
            first_evicted = n->Tokens().front();
        }
    });

    const int32_t initial = device_alloc_->AvailablePages();
    bool ok = lru_cache.EnsureCapacityByEvict<ResourceType::Device>(initial + 1);
    EXPECT_TRUE(ok);
    EXPECT_EQ(first_evicted, 1);
}

TEST(EvictionLPBDeterminism, EvictionDeterministicOnPriorityTies) {
    constexpr int32_t kPageSize = 4;
    constexpr int32_t kTotalPages = 32;
    PageAllocator alloc(kPageSize, kTotalPages);

    EvictionConfig cfg;
    cfg.policy = EvictionPolicy::kLru;
    cfg.kv_bytes_per_page = 1024;

    auto ts = std::chrono::steady_clock::now();
    TreeNode root;

    auto t1 = MakeAlignedTokens(1, kPageSize, 1);
    auto first = std::make_unique<TreeNode>(t1, ts);
    first->AttachResource<ResourceType::Device>(std::make_unique<DeviceResource>(alloc.Allocate(1)));
    TreeNode* first_raw = first.get();
    root.AddChild(t1, std::move(first));

    auto t2 = MakeAlignedTokens(1, kPageSize, 5);
    auto second = std::make_unique<TreeNode>(t2, ts);
    second->AttachResource<ResourceType::Device>(std::make_unique<DeviceResource>(alloc.Allocate(1)));
    TreeNode* second_raw = second.get();
    root.AddChild(t2, std::move(second));

    DeviceManager mgr(&alloc, cfg);
    mgr.UpdateLeaves(first_raw);
    mgr.UpdateLeaves(second_raw);

    auto evicted = mgr.Evict(1);
    ASSERT_EQ(evicted.size(), 1u);
    EXPECT_EQ(evicted.front(), first_raw);
    (void)second_raw;
}

}  // namespace tokenspeed::test
