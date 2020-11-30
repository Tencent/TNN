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

class PadLayerTest : public LayerTest,
                     public ::testing::WithParamInterface<std::tuple<int, int, int, int, int, int, int, float>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, PadLayerTest,
                         ::testing::Combine(BASIC_BATCH_CHANNEL_SIZE,
                                            // pad_w
                                            testing::Values(0, 1, 2),
                                            // pad_h
                                            testing::Values(0, 1, 2),
                                            // pad_c
                                            testing::Values(0, 1, 2),
                                            // pad_type
                                            testing::Values(0, 1),
                                            // pad value
                                            testing::Values(-FLT_MAX, 0, 2, FLT_MAX)));

TEST_P(PadLayerTest, PadLayer) {
    // get param
    int batch      = std::get<0>(GetParam());
    int channel    = std::get<1>(GetParam());
    int input_size = std::get<2>(GetParam());
    int pad_w      = std::get<3>(GetParam());
    int pad_h      = std::get<4>(GetParam());
    int pad_c      = std::get<5>(GetParam());
    int pad_type   = std::get<6>(GetParam());
    float value    = std::get<7>(GetParam());

    // insure pad is valid
    if (pad_w >= input_size) {
        pad_w = pad_w % input_size;
    }
    if (pad_h >= input_size) {
        pad_h = pad_h % input_size;
    }
    // 目前 只有pad mode 为 const 时, 才支持在channel上进行pad
    if ((pad_type == 1 || pad_type == 2) && (pad_c != 0)) {
        GTEST_SKIP();
    }
    DeviceType dev = ConvertDeviceType(FLAGS_dt);

    if ((-FLT_MAX == value || FLT_MAX == value) && DEVICE_HUAWEI_NPU == dev) {
        GTEST_SKIP();
    }

    // param
    std::shared_ptr<PadLayerParam> param(new PadLayerParam());
    param->name  = "Pad";
    param->type  = pad_type;
    param->pads  = {pad_w, pad_w, pad_h, pad_h, pad_c, pad_c};
    param->value = value;

    // generate interpreter
    std::vector<int> input_dims = {batch, channel, input_size, input_size};
    auto interpreter            = GenerateInterpreter("Pad", {input_dims}, param);
    Run(interpreter);
}

}  // namespace TNN_NS
