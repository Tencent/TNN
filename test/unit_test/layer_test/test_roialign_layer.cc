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

class RoiAlignLayerTest
    : public LayerTest,
      public ::testing::WithParamInterface<std::tuple<int, int, int, int, int, int, int, int, float>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, RoiAlignLayerTest,
                         ::testing::Combine(BASIC_BATCH_CHANNEL_SIZE,
                                            // num_rois
                                            testing::Values(1, 4, 7, 16),
                                            // pool type 0:max 1:avg
                                            testing::Values(0, 1),
                                            // output_height
                                            testing::Values(1, 3, 8),
                                            // output_width
                                            testing::Values(1, 3, 8),
                                            // sampling_ratio
                                            testing::Values(0, 1, 2),
                                            // spatial_scale
                                            testing::Values(0.125, 0.25, 0.05)));

TEST_P(RoiAlignLayerTest, RoiAlignLayer) {
    // get param
    int batch           = std::get<0>(GetParam());
    int channel         = std::get<1>(GetParam());
    int input_size      = std::get<2>(GetParam());
    int num_rois        = std::get<3>(GetParam());
    int mode            = std::get<4>(GetParam());
    int output_height   = std::get<5>(GetParam());
    int output_width    = std::get<6>(GetParam());
    int sampling_ratio  = std::get<7>(GetParam());
    float spatial_scale = std::get<8>(GetParam());

    integer_input_max_ = batch;

    DeviceType dev = ConvertDeviceType(FLAGS_dt);

    if (DEVICE_ARM != dev) {
        GTEST_SKIP();
    }

    // param
    std::shared_ptr<RoiAlignLayerParam> param(new RoiAlignLayerParam());
    param->name           = "RoiAlign";
    param->mode           = mode;
    param->output_height  = output_height;
    param->output_width   = output_width;
    param->sampling_ratio = sampling_ratio;
    param->spatial_scale  = spatial_scale;

    // generate interpreter
    std::vector<int> input_dims         = {batch, channel, input_size, input_size};
    std::vector<int> rois_dims          = {num_rois, 4};
    std::vector<int> batch_indices_dims = {num_rois};
    std::vector<DataType> input_dtype   = {DATA_TYPE_FLOAT, DATA_TYPE_FLOAT, DATA_TYPE_INT32};
    auto interpreter =
        GenerateInterpreter("RoiAlign", {input_dims, rois_dims, batch_indices_dims}, param, nullptr, 1, input_dtype);
    Run(interpreter);
}

}  // namespace TNN_NS
