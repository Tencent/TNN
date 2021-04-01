
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

static bool TestFilter(DeviceType device_type, DataType data_type) {
    if (device_type == DEVICE_NAIVE || device_type == DEVICE_METAL)
        return true;
    return false;
}

class SqueezeLayerTest : public LayerTest,
                              public ::testing::WithParamInterface<std::tuple<int, int, int, int, std::vector<int>, DataType>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, SqueezeLayerTest,
                         ::testing::Combine(BASIC_BATCH_CHANNEL_SIZE,
                                            // dim count
                                            testing::Values(2, 3, 4, 5, 6),
                                            // axis
                                            testing::Values(std::vector<int>({0}), std::vector<int>({1}),
                                                            std::vector<int>({2}), std::vector<int>({3}),
                                                            std::vector<int>({0, 1}), std::vector<int>({0, 2}),
                                                            std::vector<int>({1, -2}),std::vector<int>({0, -1}),
                                                            std::vector<int>({0, 3}), std::vector<int>({1, 3}),
                                                            std::vector<int>({1, 2, -1})),
                                            testing::Values(DATA_TYPE_FLOAT, DATA_TYPE_HALF)));

TEST_P(SqueezeLayerTest, SqueezeLayer) {
    // get param
    int dim0         = std::get<0>(GetParam());
    int dim1         = std::get<1>(GetParam());
    int dim2         = std::get<2>(GetParam());
    int dim_count    = std::get<3>(GetParam());
    auto axes        = std::get<4>(GetParam());
    DataType dtype   = std::get<5>(GetParam());
    DeviceType dev   = ConvertDeviceType(FLAGS_dt);

    if(CheckDataTypeSkip(dtype)) {
        GTEST_SKIP();
    }

    if (!TestFilter(dev, dtype)) {
        GTEST_SKIP();
    }

    // param
    std::shared_ptr<SqueezeLayerParam> param(new SqueezeLayerParam());
    param->name        = "Squeeze";
    param->axes = axes;

    // generate interpreter
    std::vector<int> input_dims = {dim0, dim1};
    while(input_dims.size() < dim_count) input_dims.push_back(dim2);
    if (input_dims.size() - axes.size() < 2) {
        GTEST_SKIP();
    }

    int erased_axes = 0;
    for(int i=axes.size()-1; i>=0; --i) {
        int axis = axes[i];
        axis = axis >= 0 ? axis : axis + input_dims.size() - erased_axes;
        if (axis >= input_dims.size() || axis < 0) {
            GTEST_SKIP();
        }
        input_dims[axis] = 1;
        erased_axes += 1;
    }
    auto interpreter            = GenerateInterpreter("Squeeze", {input_dims}, param);

    Precision precision = PRECISION_AUTO;
    if (DATA_TYPE_BFP16 == dtype) {
        precision = PRECISION_LOW;
    }
    Run(interpreter);
}

}  // namespace TNN_NS
