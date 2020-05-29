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

class InnerProductLayerTest : public LayerTest,
                              public ::testing::WithParamInterface<std::tuple<int, int, int, int, int, DataType>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, InnerProductLayerTest,
                         ::testing::Combine(testing::Values(1), testing::Values(1, 3, 10, 32),
                                            testing::Values(9, 10, 16, 19),
                                            // output channel
                                            testing::Values(21, 50),
                                            // has bias Values(0, 1)));
                                            testing::Values(0, 1), testing::Values(DATA_TYPE_FLOAT, DATA_TYPE_BFP16)));

TEST_P(InnerProductLayerTest, InnerProductLayer) {
    // get param
    int batch          = std::get<0>(GetParam());
    int input_channel  = std::get<1>(GetParam());
    int input_size     = std::get<2>(GetParam());
    int output_channel = std::get<3>(GetParam());
    int has_bias       = std::get<4>(GetParam());
    DataType dtype     = std::get<5>(GetParam());
    DeviceType dev     = ConvertDeviceType(FLAGS_dt);
    if (dtype != DATA_TYPE_FLOAT && (DEVICE_METAL == dev || DEVICE_OPENCL == dev)) {
        GTEST_SKIP();
    }

    // blob desc
    auto inputs_desc  = CreateInputBlobsDesc(batch, input_channel, input_size, 1, dtype);
    auto outputs_desc = CreateOutputBlobsDesc(1, dtype);

    // param
    InnerProductLayerParam param;
    param.name       = "InnerProduct";
    param.num_output = output_channel;
    param.has_bias   = has_bias;
    param.axis       = 1;

    // resource
    InnerProductLayerResource resource;
    int filter_count = output_channel * input_channel * input_size * input_size;
    RawBuffer filter(filter_count * sizeof(float));
    float* filter_data = filter.force_to<float*>();
    RawBuffer bias(output_channel * sizeof(float));
    float* bias_data = bias.force_to<float*>();
    InitRandom(filter_data, filter_count, 1.0f);
    InitRandom(bias_data, output_channel, 1.0f);
    resource.weight_handle = filter;
    resource.bias_handle   = bias;

    Run(LAYER_INNER_PRODUCT, &param, &resource, inputs_desc, outputs_desc);
}

}  // namespace TNN_NS
