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
#include "test/unit_test/unit_test_macro.h"

namespace TNN_NS {

class PixelShuffleLayerTest : public LayerTest, public ::testing::WithParamInterface<std::tuple<int, int, int, int>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, PixelShuffleLayerTest,
                         ::testing::Combine(testing::Values(1, 2), testing::Values(1, 2, 3, 4, 9, 10, 16, 18, 32, 50),
                                            testing::Values(9, 10, 16, 19),
                                            // upscale_factor
                                            testing::Values(1, 2, 3, 4, 5)));

TEST_P(PixelShuffleLayerTest, PixelShuffleLayer) {
    int batch          = std::get<0>(GetParam());
    int channel        = std::get<1>(GetParam());
    int input_size     = std::get<2>(GetParam());
    int upscale_factor = std::get<3>(GetParam());
    if (channel < upscale_factor || channel % (upscale_factor * upscale_factor) != 0) {
        GTEST_SKIP();
    }

    DeviceType dev = ConvertDeviceType(FLAGS_dt);

    if (DEVICE_HUAWEI_NPU == dev) {
        GTEST_SKIP();
    }

    if (DEVICE_APPLE_NPU == dev) {
        GTEST_SKIP();
    }

    std::shared_ptr<PixelShuffleLayerParam> param(new PixelShuffleLayerParam());
    param->name           = "PixelShuffle";
    param->upscale_factor = upscale_factor;

    // generate interpreter
    std::vector<int> input_dims = {batch, channel, input_size, input_size};
    auto interpreter            = GenerateInterpreter("PixelShuffle", {input_dims}, param);
    Run(interpreter);
}

}  // namespace TNN_NS
