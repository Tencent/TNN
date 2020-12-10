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

class SoftmaxLayerTest : public LayerTest,
                         public ::testing::WithParamInterface<std::tuple<int, int, int, int, int, DataType>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, SoftmaxLayerTest,
                         ::testing::Combine(testing::Values(1), testing::Values(10, 12, 10, 12, 512),
                                            testing::Values(10, 512), testing::Values(10, 512),
                                            // axis
                                            testing::Values(1, 2),
                                            // dtype
                                            testing::Values(DATA_TYPE_FLOAT)));

TEST_P(SoftmaxLayerTest, SoftmaxLayer) {
    // get param
    int batch          = std::get<0>(GetParam());
    int channel        = std::get<1>(GetParam());
    int input_height   = std::get<2>(GetParam());
    int input_width    = std::get<3>(GetParam());
    int axis           = std::get<4>(GetParam());
    DataType data_type = std::get<5>(GetParam());
    DeviceType dev     = ConvertDeviceType(FLAGS_dt);

    if (data_type == DATA_TYPE_INT8 && DEVICE_ARM != dev) {
        GTEST_SKIP();
    }

    if (1 != axis && DEVICE_HUAWEI_NPU == dev) {
        GTEST_SKIP();
    }

    if (channel < 2) {
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

    // generate interpreter
    std::vector<int> input_dims = {batch, channel, input_height, input_width};
    auto interpreter            = GenerateInterpreter("Softmax", {input_dims}, param);
    Run(interpreter);
}

}  // namespace TNN_NS
