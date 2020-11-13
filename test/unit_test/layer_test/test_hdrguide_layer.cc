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

namespace TNN_NS {

class HdrGuideLayerTest : public LayerTest, public ::testing::WithParamInterface<std::tuple<int, int, int>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, HdrGuideLayerTest, ::testing::Combine(BASIC_BATCH_CHANNEL_SIZE));

TEST_P(HdrGuideLayerTest, HdrGuideLayer) {
    // get param
    int batch      = std::get<0>(GetParam());
    int channel    = std::get<1>(GetParam());
    int input_size = std::get<2>(GetParam());
    DeviceType dev = ConvertDeviceType(FLAGS_dt);

    if (channel != 3) {
        GTEST_SKIP();
    }

    if (DEVICE_METAL != dev && DEVICE_OPENCL != dev) {
        GTEST_SKIP();
    }

    // param
    std::shared_ptr<LayerParam> param(new LayerParam());
    param->name = "HDRGuide";

    // generate interpreter
    std::vector<int> input_dims = {batch, channel, input_size, input_size};
    auto interpreter            = GenerateInterpreter("HDRGuide", {input_dims}, param);
    Run(interpreter);
}

}  // namespace TNN_NS
