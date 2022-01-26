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
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

class ShuffleLayerTest : public LayerTest, public ::testing::WithParamInterface<std::tuple<int, int, int, int, int>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, ShuffleLayerTest,
                         ::testing::Combine(BASIC_BATCH_CHANNEL_SIZE,
                                            // dim count
                                            testing::Values(2, 3, 4, 5, 6),
                                            // group
                                            testing::Values(1, 2, 3)));

TEST_P(ShuffleLayerTest, ShuffleLayer) {
    // get param
    int batch             = std::get<0>(GetParam());
    int channel_per_group = std::get<1>(GetParam());
    int input_size        = std::get<2>(GetParam());
    int dim_count         = std::get<3>(GetParam());
    int group             = std::get<4>(GetParam());
    int channel           = channel_per_group * group;

    DeviceType dev = ConvertDeviceType(FLAGS_dt);

    if (DEVICE_OPENCL == dev && dim_count > 4) {
        GTEST_SKIP();
    }
    if (DEVICE_HUAWEI_NPU == dev && dim_count != 4) {
        GTEST_SKIP();
    }
    if (DEVICE_APPLE_NPU == dev && dim_count > 5) {
        GTEST_SKIP();
    }

    // param
    std::shared_ptr<ShuffleLayerParam> param(new ShuffleLayerParam());
    param->name  = "ShuffleChannel";
    param->group = group;

    // generate interpreter
    std::vector<int> input_dims = {batch, channel};
    while(input_dims.size() < dim_count) input_dims.push_back(input_size);
    auto interpreter            = GenerateInterpreter("ShuffleChannel", {input_dims}, param);
    Run(interpreter);
}

}  // namespace TNN_NS
