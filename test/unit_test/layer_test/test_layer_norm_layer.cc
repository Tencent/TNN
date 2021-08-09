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

class LayerNormLayerTest : public LayerTest, public ::testing::WithParamInterface<std::tuple<int, int, int, int, int>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, LayerNormLayerTest,
                         ::testing::Combine(testing::Values(1, 2),              // batch
                                            testing::Values(1, 4, 6),           // channel
                                            testing::Values(10, 20, 65, 128),   // input_size
                                            testing::Values(1, 2, 3, 4, 5),     // reduce_dims_size
                                            testing::Values(2, 3, 4, 5)));      // dim count

TEST_P(LayerNormLayerTest, LayerNormLayer) {
    // get param
    int batch      = std::get<0>(GetParam());
    int channel    = std::get<1>(GetParam());
    int input_size = std::get<2>(GetParam());
    int reduce_dims_size = std::get<3>(GetParam());
    int dim_count  = std::get<4>(GetParam());
    DeviceType dev = ConvertDeviceType(FLAGS_dt);

    const int channel_dim_size = dim_count - reduce_dims_size;
    if (channel_dim_size < 0 || channel_dim_size > dim_count) {
        GTEST_SKIP();
    }

    if (DEVICE_OPENCL == dev || DEVICE_METAL == dev || DEVICE_HUAWEI_NPU == dev) {
        GTEST_SKIP();
    }

    // param
    std::shared_ptr<LayerNormLayerParam> param(new LayerNormLayerParam());
    param->name = "LayerNorm";
    param->reduce_dims_size = reduce_dims_size;

    // generate interpreter
    std::vector<int> input_dims = {batch, channel};
    while(input_dims.size() < dim_count) input_dims.push_back(input_size);

    std::vector<int> scale_bias_dims(reduce_dims_size, 0);
    for(int i=0; i<reduce_dims_size; ++i) scale_bias_dims[i] = input_dims[channel_dim_size+i];

    auto interpreter            = GenerateInterpreter("LayerNorm", {input_dims, scale_bias_dims, scale_bias_dims}, param);
    Run(interpreter);
}

}  // namespace TNN_NS
