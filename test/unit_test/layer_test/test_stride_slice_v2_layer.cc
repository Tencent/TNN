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

class StrideSliceV2LayerTest
    : public LayerTest,
      public ::testing::WithParamInterface<std::tuple<int, int, int, int, int, int, int, int, DataType>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, StrideSliceV2LayerTest,
                         ::testing::Combine(testing::Values(1, 2),
                                            // channel
                                            testing::Values(3, 4, 7, 8, 15),
                                            // dim
                                            testing::Values(3, 6),
                                            // dims count
                                            testing::Values(2, 3, 4, 5),
                                            // start
                                            testing::Values(0, 1),
                                            // end
                                            testing::Values(3, INT_MAX),
                                            // axis
                                            testing::Values(1, 2, 3, 4),
                                            // step
                                            testing::Values(1, 2), testing::Values(DATA_TYPE_FLOAT, DATA_TYPE_HALF)));

TEST_P(StrideSliceV2LayerTest, StrideSliceV2Layer) {
    // get param
    int batch          = std::get<0>(GetParam());
    int channel        = std::get<1>(GetParam());
    int dim            = std::get<2>(GetParam());
    int dims_count     = std::get<3>(GetParam());
    int start          = std::get<4>(GetParam());
    int end            = std::get<5>(GetParam());
    int axis           = std::get<6>(GetParam());
    int step           = std::get<7>(GetParam());
    DataType data_type = std::get<8>(GetParam());
    if (axis >= dims_count) {
        GTEST_SKIP();
    }
    DeviceType dev        = ConvertDeviceType(FLAGS_dt);
    Precision precision   = SetPrecision(dev, data_type);
    DimsVector input_dims = {batch, channel};
    for (int i = 2; i < dims_count; ++i) {
        input_dims.push_back(dim);
    }
    // param
    std::shared_ptr<StrideSliceV2LayerParam> param(new StrideSliceV2LayerParam());
    param->name    = "StrideSliceV2";
    param->begins  = {start};
    param->ends    = {end};
    param->axes    = {axis};
    param->strides = {step};
    // generate interpreter
    auto interpreter = GenerateInterpreter("StridedSliceV2", {input_dims}, param);

    Run(interpreter, precision);
}

}  // namespace TNN_NS
