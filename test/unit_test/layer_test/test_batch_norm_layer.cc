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

class BatchNormScaleLayerTest : public LayerTest,
                                public ::testing::WithParamInterface<std::tuple<int, int, int, int, bool, bool, DataType>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, BatchNormScaleLayerTest,
                         ::testing::Combine(BASIC_BATCH_CHANNEL_SIZE,
                                            // dim count
                                            testing::Values(2, 3, 4, 5),
                                            // share channel
                                            testing::Values(false, true),
                                            // has bias
                                            testing::Values(true, false),
                                            testing::Values(DATA_TYPE_FLOAT, DATA_TYPE_HALF)));

TEST_P(BatchNormScaleLayerTest, BatchNormScaleLayer) {
    // get param
    int batch          = std::get<0>(GetParam());
    int channel        = std::get<1>(GetParam());
    int input_size     = std::get<2>(GetParam());
    int dim_count      = std::get<3>(GetParam());
    bool share_channel = std::get<4>(GetParam());
    bool has_bias      = std::get<5>(GetParam());
    auto dtype         = std::get<6>(GetParam());

    DeviceType dev = ConvertDeviceType(FLAGS_dt);

    if(CheckDataTypeSkip(dtype)) {
        GTEST_SKIP();
    }

    if (dim_count != 4) {
        GTEST_SKIP();
    }

    // param
    std::shared_ptr<BatchNormLayerParam> param(new BatchNormLayerParam());
    param->name = "BatchNorm";
    param->channels = channel;

    // resource
    std::shared_ptr<BatchNormLayerResource> resource(new BatchNormLayerResource());
    int k_count = share_channel ? 1 : channel;
    RawBuffer filter_k(k_count * sizeof(float));
    float* k_data = filter_k.force_to<float*>();
    InitRandom(k_data, k_count, 1.0f);
    resource->scale_handle = filter_k;
    if (has_bias) {
        RawBuffer bias(k_count * sizeof(float));
        float* bias_data = bias.force_to<float*>();
        InitRandom(bias_data, k_count, 1.0f);
        resource->bias_handle = bias;
    }

    // generate interpreter
    //std::vector<int> input_dims = {batch, channel, input_size, input_size};
    std::vector<int> input_dims = {batch, channel};
    while(input_dims.size() < dim_count) input_dims.push_back(input_size);
    auto interpreter1 = GenerateInterpreter("BatchNormCxx", {input_dims}, param, resource);
    Precision precision = SetPrecision(dev, dtype);
    Run(interpreter1, precision);

    if (DEVICE_APPLE_NPU != dev) {
        auto interpreter2 = GenerateInterpreter("Scale", {input_dims}, param, resource);
        Run(interpreter2, precision);
    }
}

}  // namespace TNN_NS
