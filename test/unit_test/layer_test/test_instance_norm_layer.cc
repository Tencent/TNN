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

class InstanceNormLayerTest : public LayerTest,
                              public ::testing::WithParamInterface<std::tuple<int, int, int, int, DataType>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, InstanceNormLayerTest,
                         ::testing::Combine(testing::Values(1, 2), testing::Values(1, 4, 6),
                                            testing::Values(10, 20, 65, 128),
                                            // dim count
                                            testing::Values(3, 4, 5),
                                            testing::Values(DATA_TYPE_FLOAT, DATA_TYPE_HALF)));

TEST_P(InstanceNormLayerTest, InstanceNormLayer) {
    // get param
    int batch      = std::get<0>(GetParam());
    int channel    = std::get<1>(GetParam());
    int input_size = std::get<2>(GetParam());
    int dim_count  = std::get<3>(GetParam());
    DataType dtype = std::get<4>(GetParam());
    DeviceType dev = ConvertDeviceType(FLAGS_dt);
    if (input_size > 64 && dim_count > 5) {
        GTEST_SKIP();
    }

    if (DEVICE_OPENCL == dev & dim_count > 4) {
        GTEST_SKIP();
    }
    if (DEVICE_HUAWEI_NPU == dev & dim_count != 4) {
        GTEST_SKIP();
    }
    if (CheckDataTypeSkip(dtype)) {
        GTEST_SKIP();
    }
    // param
    std::shared_ptr<InstanceNormLayerParam> param(new InstanceNormLayerParam());
    param->name = "InstanceNorm";
    param->channels = channel;

    // resource
    std::shared_ptr<InstanceNormLayerResource> resource(new InstanceNormLayerResource());
    RawBuffer filter_k(channel * sizeof(float));
    float* k_data = filter_k.force_to<float*>();
    InitRandom(k_data, channel, 1.0f);
    resource->scale_handle = filter_k;
    RawBuffer bias(channel * sizeof(float));
    float* bias_data = bias.force_to<float*>();
    InitRandom(bias_data, channel, 1.0f);
    resource->bias_handle = bias;

    // generate interpreter
    std::vector<int> input_dims = {batch, channel};
    while (input_dims.size() < dim_count) {
        input_dims.push_back(input_size);
    }
    auto interpreter    = GenerateInterpreter("InstBatchNormCxx", {input_dims}, param, resource);
    Precision precision = SetPrecision(dev, dtype);
    Run(interpreter, precision);
}

}  // namespace TNN_NS
