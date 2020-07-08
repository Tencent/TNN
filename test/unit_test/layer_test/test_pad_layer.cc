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
                     public ::testing::WithParamInterface<std::tuple<int, int, int, int, int, int, float>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, PadLayerTest,
                         ::testing::Combine(BASIC_BATCH_CHANNEL_SIZE,
                                            // pad_l
                                            testing::Values(0, 1, 2),
                                            // pad_t
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
    int pad_l      = std::get<3>(GetParam());
    int pad_t      = std::get<4>(GetParam());
    int pad_type   = std::get<5>(GetParam());
    float value    = std::get<6>(GetParam());

    // insure pad is valid
    if (pad_l >= input_size) {
        pad_l = pad_l % input_size;
    }
    if (pad_t >= input_size) {
        pad_t = pad_t % input_size;
    }
    DeviceType dev = ConvertDeviceType(FLAGS_dt);

    // blob desc
    auto inputs_desc  = CreateInputBlobsDesc(batch, channel, input_size, 1, DATA_TYPE_FLOAT);
    auto outputs_desc = CreateOutputBlobsDesc(1, DATA_TYPE_FLOAT);

    // param
    PadLayerParam param;
    param.name = "Pad";
    param.type = pad_type;
    param.pads = {pad_l, pad_l, pad_t, pad_t};
    param.value = value;

    Run(LAYER_PAD, &param, nullptr, inputs_desc, outputs_desc);
}

}  // namespace TNN_NS
