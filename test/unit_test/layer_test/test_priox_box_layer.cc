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

class PriorBoxLayerTest
    : public LayerTest,
      public ::testing::WithParamInterface<
          std::tuple<int, int, int, float, float, bool, bool, std::vector<float>, std::vector<float>, int, float>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, PriorBoxLayerTest,
                         ::testing::Combine(BASIC_BATCH_CHANNEL_SIZE, ::testing::Values(1.0, 2.0),
                                            ::testing::Values(8.0, 10.0), ::testing::Values(false, true),
                                            ::testing::Values(true, false),
                                            ::testing::Values(std::vector<float>({0.1, 0.1, 0.2, 0.2})),
                                            ::testing::Values(std::vector<float>({1.0, 2.0, 3.0, 0.5, 0.33333})),
                                            ::testing::Values(512, 1024), ::testing::Values(1.0)));

TEST_P(PriorBoxLayerTest, PriorBoxLayer) {
    // get param
    int batch                        = std::get<0>(GetParam());
    int channel                      = std::get<1>(GetParam());
    int input_size                   = std::get<2>(GetParam());
    float min_size                   = std::get<3>(GetParam());
    float max_size                   = std::get<4>(GetParam());
    bool clip                        = std::get<5>(GetParam());
    bool flip                        = std::get<6>(GetParam());
    std::vector<float> variances     = std::get<7>(GetParam());
    std::vector<float> aspect_ratios = std::get<8>(GetParam());
    int img_size                     = std::get<9>(GetParam());
    int step_size                    = std::get<10>(GetParam());

    DeviceType dev = ConvertDeviceType(FLAGS_dt);

    if (DEVICE_HUAWEI_NPU == dev && batch != 1) {
        GTEST_SKIP();
    }

    std::vector<float> min_sizes = {min_size};
    std::vector<float> max_sizes = {max_size};
    float offset                 = 0.5;

    std::shared_ptr<PriorBoxLayerParam> param(new PriorBoxLayerParam());
    param->name          = "PriorBox";
    param->min_sizes     = min_sizes;
    param->max_sizes     = max_sizes;
    param->clip          = clip;
    param->flip          = flip;
    param->variances     = variances;
    param->aspect_ratios = aspect_ratios;
    param->img_w = param->img_h = img_size;
    param->step_h = param->step_w = step_size;
    param->offset                 = offset;

    // generate interpreter
    std::vector<int> input_dims = {batch, channel, input_size, input_size};
    auto interpreter            = GenerateInterpreter("PriorBox", {input_dims}, param);
    Run(interpreter);
}

}  // namespace TNN_NS
