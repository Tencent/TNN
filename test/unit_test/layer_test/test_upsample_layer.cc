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

class UpsampleLayerTest
    : public LayerTest,
      public ::testing::WithParamInterface<std::tuple<int, int, int, int, int, float, float, bool, DataType>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, UpsampleLayerTest,
                         ::testing::Combine(BASIC_BATCH_CHANNEL_SIZE,
                                            // resize type 1:nearest 2:bilinear
                                            testing::Values(1, 2),
                                            // align_corners
                                            testing::Values(0, 1),
                                            // scale x Values(1.0, 1.45, 2, 2.78)
                                            testing::Values(0.3, 0.5, 1.0, 1.45, 2, 2.78),
                                            // scale y Values(1.0, 1.45, 2, 2.78)
                                            testing::Values(0.3, 0.5, 1.0, 1.45, 2, 2.78),
                                            // use dims
                                            testing::Values(true, false),
                                            // data_type
                                            testing::Values(DATA_TYPE_FLOAT, DATA_TYPE_INT8)));

TEST_P(UpsampleLayerTest, UpsampleLayer) {
    // get param
    int batch         = std::get<0>(GetParam());
    int channel       = std::get<1>(GetParam());
    int input_size    = std::get<2>(GetParam());
    int mode          = std::get<3>(GetParam());
    int align_corners = std::get<4>(GetParam());
    float scale_x     = std::get<5>(GetParam());
    float scale_y     = std::get<6>(GetParam());
    bool use_dims     = std::get<7>(GetParam());
    auto data_type    = std::get<8>(GetParam());

    DeviceType dev = ConvertDeviceType(FLAGS_dt);

    if (data_type == DATA_TYPE_INT8 && DEVICE_ARM != dev) {
        GTEST_SKIP();
    }

    if (DEVICE_HUAWEI_NPU == dev && (mode == 2 || ((int)scale_x != scale_x || (int)scale_y != scale_y))) {
        GTEST_SKIP();
    }

    // param
    std::shared_ptr<UpsampleLayerParam> param(new UpsampleLayerParam());
    param->name          = "Upsample";
    param->mode          = mode;
    param->align_corners = align_corners;
    param->scales        = {scale_x, scale_y};
    if (use_dims) {
        param->dims = {(int)(scale_x * input_size), (int)(scale_y * input_size)};
    }

    // generate interpreter
    std::vector<int> input_dims = {batch, channel, input_size, input_size};
    auto interpreter            = GenerateInterpreter("Upsample", {input_dims}, param);
    Run(interpreter);
}

}  // namespace TNN_NS
