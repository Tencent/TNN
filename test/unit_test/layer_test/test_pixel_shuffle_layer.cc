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
#include "test/unit_test/unit_test_macro.h"

namespace TNN_NS {

class PixelShuffleLayerTest : public LayerTest, public ::testing::WithParamInterface<std::tuple<int, int, int, int>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, PixelShuffleLayerTest,
                         ::testing::Combine(BASIC_BATCH_CHANNEL_SIZE,
                                            // upscale_factor
                                            testing::Values(1, 2, 3)));

TEST_P(PixelShuffleLayerTest, PixelShuffleLayer) {
    int batch          = std::get<0>(GetParam());
    int channel        = std::get<1>(GetParam());
    int input_size     = std::get<2>(GetParam());
    int upscale_factor = std::get<3>(GetParam());
    if (channel < upscale_factor || channel % (upscale_factor * upscale_factor) != 0) {
        GTEST_SKIP();
    }

    DeviceType dev = ConvertDeviceType(FLAGS_dt);

    // blob desc
    auto inputs_desc  = CreateInputBlobsDesc(batch, channel, input_size, 1, DATA_TYPE_FLOAT);
    auto outputs_desc = CreateOutputBlobsDesc(1, DATA_TYPE_FLOAT);

    PixelShuffleLayerParam param;
    param.name           = "PixelShuffle";
    param.upscale_factor = upscale_factor;

    Run(LAYER_PIXEL_SHUFFLE, &param, nullptr, inputs_desc, outputs_desc);
}

}  // namespace TNN_NS