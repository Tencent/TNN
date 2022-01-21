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

class ReshapeLayerTest : public LayerTest,
                         public ::testing::WithParamInterface<std::tuple<int, int, int, int, int, DataType>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, ReshapeLayerTest,
                         ::testing::Combine(BASIC_BATCH_CHANNEL_SIZE,
                                            // dimensions
                                            testing::Values(2, 3, 4, 5, 6),
                                            // reshape type
                                            testing::Values(0, 1),
                                            testing::Values(DATA_TYPE_FLOAT, DATA_TYPE_HALF, DATA_TYPE_BFP16,
                                                            DATA_TYPE_INT8)));

TEST_P(ReshapeLayerTest, ReshapeLayer) {
    // get param
    int batch        = std::get<0>(GetParam());
    int channel      = std::get<1>(GetParam());
    int input_size   = std::get<2>(GetParam());
    int dim_size     = std::get<3>(GetParam());
    int reshape_type = std::get<4>(GetParam());
    auto data_type   = std::get<5>(GetParam());
    DeviceType dev   = ConvertDeviceType(FLAGS_dt);

    if (0 != reshape_type && DEVICE_HUAWEI_NPU == dev) {
        GTEST_SKIP();
    }
    if (dim_size > 4 && DEVICE_HUAWEI_NPU == dev) {
        GTEST_SKIP();
    }

    if (DATA_TYPE_INT8 == data_type && (dev != DEVICE_ARM && dev != DEVICE_NAIVE)) {
        GTEST_SKIP();
    }

    // APPLE_NPU can not support dim > 5
    if ((dim_size > 5)&& DEVICE_APPLE_NPU == dev) {
        GTEST_SKIP();
    }
    
    // reshape_type 1 does not support dims>4
    if (reshape_type == 1 && dim_size > 4) {
        GTEST_SKIP();
    }

    std::vector<int> input_dims = {batch, channel};
    while (input_dims.size() < dim_size) {
        input_dims.push_back(input_size);
    }

    // param
    std::shared_ptr<ReshapeLayerParam> param(new ReshapeLayerParam());
    param->name         = "Reshape";
    param->reshape_type = reshape_type;
    param->axis         = 0;
    param->num_axes     = dim_size;
    param->shape        = {0, -1};
    while (param->shape.size() < dim_size) {
        param->shape.push_back(1);
    }
    if (DATA_TYPE_INT8 == data_type) {
        param->quantized = true;
    }
    Precision precision = SetPrecision(dev, data_type);

    // generate interpreter
    auto interpreter = GenerateInterpreter("Reshape", {input_dims}, param);
    Run(interpreter, precision);
}

}  // namespace TNN_NS
