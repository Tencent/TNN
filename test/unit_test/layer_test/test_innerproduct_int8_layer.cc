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

class InnerProductInt8LayerTest : public LayerTest,
                                  public ::testing::WithParamInterface<std::tuple<int, int, int>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, InnerProductInt8LayerTest,
                         ::testing::Combine(testing::Values(1), testing::Values(3, 4, 8, 9, 16),
                                            // output channel
                                            testing::Values(1, 4, 8, 16, 32)));

TEST_P(InnerProductInt8LayerTest, InnerProductLayer) {
    // get param
    int batch          = std::get<0>(GetParam());
    int input_channel  = std::get<1>(GetParam());
    int output_channel = std::get<2>(GetParam());
    int input_size     = 1;
    DeviceType dev     = ConvertDeviceType(FLAGS_dt);
    if (DEVICE_ARM != dev) {
        GTEST_SKIP();
    }

    // blob desc
    auto inputs_desc  = CreateInputBlobsDesc(batch, input_channel, input_size, 1, DATA_TYPE_INT8);
    auto outputs_desc = CreateOutputBlobsDesc(1, DATA_TYPE_INT8);
    // assign output dims to ensure output resourse be created correctly
    outputs_desc[0].dims.push_back(batch);
    outputs_desc[0].dims.push_back(output_channel);
    outputs_desc[0].dims.push_back(1);
    outputs_desc[0].dims.push_back(1);

    // param
    InnerProductLayerParam param;
    param.name       = "InnerProduct";
    param.num_output = output_channel;
    param.has_bias   = 0;
    param.axis       = 1;

    // resource
    InnerProductLayerResource resource;
    size_t filter_count = output_channel * input_channel * input_size * input_size;
    RawBuffer filter(filter_count * sizeof(int8_t));
    int8_t* filter_data = filter.force_to<int8_t*>();
    InitRandom<int8_t>(filter_data, filter_count, 4);
    RawBuffer scale(output_channel * sizeof(float));
    InitRandom(scale.force_to<float*>(), output_channel, 1.0f);
    for (int i = 0; i < output_channel; i++) {
        scale.force_to<float*>()[i] =
            std::fabs(scale.force_to<float*>()[i] - 0.f) < FLT_EPSILON ? 1.f : scale.force_to<float*>()[i];
    }
    resource.weight_handle = filter;
    resource.weight_handle.SetDataType(DATA_TYPE_INT8);
    resource.scale_handle = scale;
    Run(LAYER_INNER_PRODUCT, &param, &resource, inputs_desc, outputs_desc);
}

}  // namespace TNN_NS
