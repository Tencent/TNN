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
#include "tnn/utils/cpu_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

class GridSampleLayerTest
    : public LayerTest,
      public ::testing::WithParamInterface<std::tuple<int, int, int, int, int, int, int, int, int, DataType>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, GridSampleLayerTest,
                         ::testing::Combine(testing::Values(1, 2),
                                            // channel
                                            testing::Values(1, 3, 4, 10),
                                            // input_height
                                            testing::Values(3, 4, 10),
                                            // input_weight
                                            testing::Values(3, 4, 10),
                                            // output_height
                                            testing::Values(3, 4, 10, 20),
                                            // output_weight
                                            testing::Values(3, 4, 10, 20),
                                            // mode:
                                            testing::Values(2),
                                            // pad_type
                                            testing::Values(0),
                                            // align_corners
                                            testing::Values(0),
                                            // dtype
                                            testing::Values(DATA_TYPE_FLOAT)));

TEST_P(GridSampleLayerTest, GridSampleLayer) {
    // get param
    int batch          = std::get<0>(GetParam());
    int channel        = std::get<1>(GetParam());
    int input_height   = std::get<2>(GetParam());
    int input_width    = std::get<3>(GetParam());
    int output_height  = std::get<4>(GetParam());
    int output_width   = std::get<5>(GetParam());
    int mode           = std::get<6>(GetParam());
    int pad_type       = std::get<7>(GetParam());
    int align_corners  = std::get<8>(GetParam());
    DataType data_type = std::get<9>(GetParam());
    DeviceType dev     = ConvertDeviceType(FLAGS_dt);

    if (CheckDataTypeSkip(data_type)) {
        GTEST_SKIP();
    }
    if (!(DEVICE_NAIVE == dev || DEVICE_ARM == dev || DEVICE_CUDA == dev || DEVICE_OPENCL == dev)) {
        GTEST_SKIP();
    }

    if (!(mode == 2 && pad_type == 0 && align_corners == 0)) {
        GTEST_SKIP();
    }

    Precision precision = SetPrecision(dev, data_type);

    // param
    std::shared_ptr<GridSampleLayerParam> param(new GridSampleLayerParam());
    param->name          = "GridSample";
    param->mode          = mode;
    param->pad_type      = pad_type;
    param->align_corners = align_corners;

    // generate interpreter
    std::vector<std::vector<int32_t>> input_dims_vec;
    std::vector<int32_t> input_dims = {batch, channel, input_height, input_width};
    std::vector<int32_t> grid_dims  = {batch, output_height, output_width, 2};
    input_dims_vec.push_back(input_dims);
    input_dims_vec.push_back(grid_dims);
    auto interpreter = GenerateInterpreter("GridSample", input_dims_vec, param);
    Run(interpreter, precision);
}

}  // namespace TNN_NS
