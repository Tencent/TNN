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
#include "tnn/utils/cpu_utils.h"

namespace TNN_NS {

class SoftmaxLayerTest : public LayerTest,
                         public ::testing::WithParamInterface<std::tuple<int, int, int, int, int, int, DataType>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, SoftmaxLayerTest,
                         ::testing::Combine(testing::Values(1, 2), testing::Values(10, 12, 512),
                                            testing::Values(10, 512), testing::Values(1, 10, 512),
                                            // axis
                                            testing::Values(-1, 0, 1, 2, 3, 4),
                                            // dim count
                                            testing::Values(2, 3, 4, 5),
                                            // dtype
                                            testing::Values(DATA_TYPE_FLOAT, DATA_TYPE_HALF, DATA_TYPE_BFP16)));

TEST_P(SoftmaxLayerTest, SoftmaxLayer) {
    // get param
    int batch          = std::get<0>(GetParam());
    int channel        = std::get<1>(GetParam());
    int input_height   = std::get<2>(GetParam());
    int input_width    = std::get<3>(GetParam());
    int axis           = std::get<4>(GetParam());
    int dim_count      = std::get<5>(GetParam());
    DataType data_type = std::get<6>(GetParam());
    DeviceType dev     = ConvertDeviceType(FLAGS_dt);

    if(CheckDataTypeSkip(data_type)) {
        GTEST_SKIP();
    }
    
    if (dev == DEVICE_ARM && axis == 0) {
        // arm do not support axis == 0 now
        GTEST_SKIP();
    }

    if (DEVICE_OPENCL == dev && (dim_count > 4 || (axis != 1 && axis != 2 && axis != 3))) {
        // opencl only support axis = 1, 2 or 3 for now
        GTEST_SKIP();
    }

    if (2 == axis && DEVICE_HUAWEI_NPU == dev) {
        GTEST_SKIP();
    }

    if (dim_count != 4 && DEVICE_HUAWEI_NPU == dev) {
        GTEST_SKIP();
    }

    if (channel < 2) {
        GTEST_SKIP();
    }

    if (axis >= dim_count) {
        GTEST_SKIP();
    }

    if ((channel == 512 && input_height == 512) || (input_width == 512 && input_height == 512) ||
        (channel == 512 && input_width == 512)) {
        GTEST_SKIP();
    }

    // param
    std::shared_ptr<SoftmaxLayerParam> param(new SoftmaxLayerParam());
    param->name = "Softmax";
    param->axis = axis;

    auto precision = SetPrecision(dev, data_type);

    // generate interpreter
    std::vector<int> input_dims = {batch, channel};
    std::vector<int> input_sizes = {input_height, input_width};
    auto idx = 0;
    while(input_dims.size() < dim_count) {
        input_dims.push_back(input_sizes[idx]);
        idx  = (idx + 1) % 2;
    }
    auto interpreter            = GenerateInterpreter("Softmax", {input_dims}, param);
    Run(interpreter, precision);
}

}  // namespace TNN_NS
