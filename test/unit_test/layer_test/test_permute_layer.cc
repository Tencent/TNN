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

#include <numeric>
#include <map>
#include "test/unit_test/layer_test/layer_test.h"
#include "test/unit_test/unit_test_common.h"
#include "test/unit_test/utils/network_helpers.h"
#include "tnn/utils/dims_utils.h"

using List = std::vector<int>;
namespace TNN_NS {
static std::vector<List>& GetPermuteOrders(int dim_size) {
    static std::map<int, std::vector<List>> PermutationLib;
    if (PermutationLib.count(dim_size) == 0) {
        List base_order(dim_size, 0);
        std::iota(base_order.begin(), base_order.end(), 0);
        std::vector<List> all_permutations;
        // initialize
        do {
            all_permutations.push_back(base_order);
        } while(std::next_permutation(base_order.begin(), base_order.end()));
        PermutationLib.insert({dim_size, all_permutations});
    }
    return PermutationLib[dim_size];
}

class PermuteLayerTest : public LayerTest, public ::testing::WithParamInterface<std::tuple<int, int, int, int>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, PermuteLayerTest,
                         ::testing::Combine(BASIC_BATCH_CHANNEL_SIZE,
                         testing::Values(2, 3, 4, 5)));

TEST_P(PermuteLayerTest, PermuteLayer) {
    // get param
    int batch      = std::get<0>(GetParam());
    int channel    = std::get<1>(GetParam());
    int input_size = std::get<2>(GetParam());
    int dim_count  = std::get<3>(GetParam());

    DeviceType dev = ConvertDeviceType(FLAGS_dt);

    if (DEVICE_HUAWEI_NPU == dev) {
        GTEST_SKIP();
    }

    // generate interpreter
    std::vector<int> input_dims = {batch, channel};
    while(input_dims.size() < dim_count) input_dims.push_back(input_size);

    for(const auto& orders : GetPermuteOrders(dim_count)) {
        // param
        std::shared_ptr<PermuteLayerParam> param(new PermuteLayerParam());
        param->orders = orders;

        auto interpreter            = GenerateInterpreter("Permute", {input_dims}, param);
        Run(interpreter);
    }
}

}  // namespace TNN_NS
