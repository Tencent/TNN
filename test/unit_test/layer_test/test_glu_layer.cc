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

class GLULayerTest : public LayerTest,
                     public ::testing::WithParamInterface<std::tuple<int, int, int, int, int, int, DataType>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, GLULayerTest,
                         ::testing::Combine(testing::Values(1, 2), testing::Values(8, 10, 16, 20),
                                            testing::Values(8, 10, 16),
                                            testing::Values(8, 10 ,16),
                                            // axis
                                            testing::Values(1, 2, 3, 4),
                                            // dim count
                                            testing::Values(3, 4, 5),
                                            // dtype
                                            testing::Values(DATA_TYPE_FLOAT, DATA_TYPE_HALF, DATA_TYPE_BFP16)));

TEST_P(GLULayerTest, GLULayer) {
    // get param
    int batch          = std::get<0>(GetParam());
    int channel        = std::get<1>(GetParam());
    int input_height   = std::get<2>(GetParam());
    int input_width    = std::get<3>(GetParam());
    int axis           = std::get<4>(GetParam());
    int dim_count      = std::get<5>(GetParam());
    DataType data_type = std::get<6>(GetParam());
    DeviceType dev     = ConvertDeviceType(FLAGS_dt);

    if (CheckDataTypeSkip(data_type)) {
        GTEST_SKIP();
    }
    if (DEVICE_ARM != dev) {
        // only arm device support glu layer
        GTEST_SKIP();
    }
    if (data_type == DATA_TYPE_BFP16) {
        GTEST_SKIP();
    }
    if (axis >= dim_count) {
        GTEST_SKIP();
    }
    // param
    std::shared_ptr<GLULayerParam> param(new GLULayerParam());
    param->name = "GLU";
    param->axis = axis;

    auto precision = SetPrecision(dev, data_type);

    // generate interpreter
    std::vector<int> input_dims  = {batch, channel};
    std::vector<int> input_sizes = {input_height, input_width};
    auto idx                     = 0;
    while (input_dims.size() < dim_count) {
        input_dims.push_back(input_sizes[idx]);
        idx = (idx + 1) % 2;
    }
    if (input_dims[axis] % 2 != 0) {
        GTEST_SKIP();
    }
    if (axis == 1 && input_dims[axis] % 8 != 0 && data_type == DATA_TYPE_FLOAT) {
        GTEST_SKIP();
    }

    if (axis == 1 && input_dims[axis] % 16 != 0 && data_type == DATA_TYPE_HALF) {
        GTEST_SKIP();
    }
    auto interpreter = GenerateInterpreter("GLU", {input_dims}, param);
    Run(interpreter, precision);
}

}  // namespace TNN_NS
