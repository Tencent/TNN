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

class SplitVLayerTest : public LayerTest,
                        public ::testing::WithParamInterface<std::tuple<int, int, int, int, int, int, DataType>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, SplitVLayerTest,
                         ::testing::Combine(BASIC_BATCH_CHANNEL_SIZE,
                                            // dim count
                                            testing::Values(2, 3, 4, 5, 6),
                                            // axis
                                            testing::Values(0, 1, 2, 3, 4, 5),
                                            // output cnt
                                            testing::Values(2, 3),
                                            // dtype
                                            testing::Values(DATA_TYPE_FLOAT)));

TEST_P(SplitVLayerTest, SplitVLayer) {
    // get param
    int batch          = std::get<0>(GetParam());
    int channel        = std::get<1>(GetParam());
    int input_size     = std::get<2>(GetParam());
    int dim_count      = std::get<3>(GetParam());
    int axis           = std::get<4>(GetParam());
    int output_count   = std::get<5>(GetParam());
    DataType data_type = std::get<6>(GetParam());
    DeviceType dev     = ConvertDeviceType(FLAGS_dt);

    if (dim_count > 4 && DEVICE_HUAWEI_NPU == dev) {
        GTEST_SKIP();
    }

    if (dim_count > 5 && DEVICE_APPLE_NPU == dev) {
        GTEST_SKIP();
    }

    if (DEVICE_OPENCL == dev && (dim_count > 4 || axis != 1)) {
        GTEST_SKIP();
    }

    std::vector<int> input_dims = {batch, channel};
    while (input_dims.size() < dim_count) input_dims.push_back(input_size);

    if (axis >= dim_count) {
        GTEST_SKIP();
    }

    if (input_dims[axis] < output_count) {
        GTEST_SKIP();
    }


    // param
    std::shared_ptr<SplitVLayerParam> param(new SplitVLayerParam());
    param->name   = "SplitV";
    param->axis   = axis;
    std::vector<int> slices;
    int sum = 0;
    for(int i=0; i<output_count; ++i) {
        if (i != output_count - 1) {
            slices.push_back(input_dims[axis] / output_count);
            sum += input_dims[axis] / output_count;
        } else {
            slices.push_back(input_dims[axis] - sum);
        }
    }
    param->slices = slices;

    // generate interpreter
    auto interpreter            = GenerateInterpreter("SplitV", {input_dims}, param, nullptr, output_count);
    Run(interpreter);
}

}  // namespace TNN_NS
