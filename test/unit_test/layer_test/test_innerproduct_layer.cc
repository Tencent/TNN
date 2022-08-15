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

class InnerProductLayerTest : public LayerTest,
                              public ::testing::WithParamInterface<std::tuple<int, int, int, int, int, DataType>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, InnerProductLayerTest,
                         ::testing::Combine(testing::Values(1, 2, 8, 11), testing::Values(1, 3, 10, 32),
                                            testing::Values(9, 10, 16, 19),
                                            // output channel
                                            testing::Values(4, 8, 21, 50),
                                            // has bias Values(0, 1)));
                                            testing::Values(0, 1),
                                            testing::Values(DATA_TYPE_FLOAT, DATA_TYPE_HALF, DATA_TYPE_BFP16)));

TEST_P(InnerProductLayerTest, InnerProductLayer) {
    // get param
    int batch          = std::get<0>(GetParam());
    int input_channel  = std::get<1>(GetParam());
    int input_size     = std::get<2>(GetParam());
    int output_channel = std::get<3>(GetParam());
    int has_bias       = std::get<4>(GetParam());
    DataType dtype     = std::get<5>(GetParam());
    DeviceType dev     = ConvertDeviceType(FLAGS_dt);

    if (dev == DEVICE_ARM && dtype == DATA_TYPE_HALF) {
        // error of fp16 result will accumulate as input size increases
        if (input_channel * input_size * input_size > 5000) {
            GTEST_SKIP();
        }
    }

    if(CheckDataTypeSkip(dtype)) {
        GTEST_SKIP();
    }

    // param
    std::shared_ptr<InnerProductLayerParam> param(new InnerProductLayerParam());
    param->name       = "InnerProduct";
    param->num_output = output_channel;
    param->has_bias   = has_bias;
    param->axis       = 1;

    // generate interpreter
    std::vector<int> input_dims = {batch, input_channel, input_size, input_size};
    //coreml innerproduct layer output rank is same as the input rank
    if (DEVICE_APPLE_NPU == dev) { 
        input_dims = {batch, input_size};
    }
    auto interpreter            = GenerateInterpreter("InnerProduct", {input_dims}, param);

    Precision precision = SetPrecision(dev, dtype);
    Run(interpreter, precision);
}

}  // namespace TNN_NS
