// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "test/unit_test/layer_test/layer_test.h"
#include "test/unit_test/unit_test_common.h"
#include "test/unit_test/utils/network_helpers.h"
#include "tnn/utils/dims_vector_utils.h"

#include <algorithm>

namespace TNN_NS {

static void GetPermuteOrders(std::vector<int>& orders, int permute_type) {
    using List = std::vector<int>;
    static List base_order = {0, 1, 2, 3};
    static std::vector<List> all_permutations;
    // initialize
    if (all_permutations.size() <= 0) {
        do {
            all_permutations.push_back(base_order);
        } while(std::next_permutation(base_order.begin(), base_order.end()));
    }

    if (permute_type < all_permutations.size()) {
        orders = all_permutations[permute_type];
    } else {
        orders = base_order;
    }
}
// 24 possible permutations for a 4-dimensional tensor
constexpr static int kPermutations = 24;

class PermuteLayerTest : public LayerTest, public ::testing::WithParamInterface<std::tuple<int, int, int, int>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, PermuteLayerTest,
                         ::testing::Combine(BASIC_BATCH_CHANNEL_SIZE, testing::Range(0, kPermutations)));

TEST_P(PermuteLayerTest, PermuteLayer) {
    // get param
    int batch      = std::get<0>(GetParam());
    int channel    = std::get<1>(GetParam());
    int input_size = std::get<2>(GetParam());
    int order_type = std::get<3>(GetParam());

    DeviceType dev = ConvertDeviceType(FLAGS_dt);

    if (DEVICE_HUAWEI_NPU == dev) {
        GTEST_SKIP();
    }

    // param
    std::shared_ptr<PermuteLayerParam> param(new PermuteLayerParam());
    GetPermuteOrders(param->orders, order_type);

    // generate interpreter
    std::vector<int> input_dims = {batch, channel, input_size, input_size};
    auto interpreter            = GenerateInterpreter("Permute", {input_dims}, param);
    Run(interpreter);
}

}  // namespace TNN_NS
