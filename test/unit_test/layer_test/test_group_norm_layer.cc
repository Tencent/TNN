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

class GroupNormLayerTest : public LayerTest,
                           public ::testing::WithParamInterface<std::tuple<int, int, int, int, int, DataType>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, GroupNormLayerTest,
                         ::testing::Combine(testing::Values(1, 2),             // batch
                                            testing::Values(1, 6, 24),         // channel
                                            testing::Values(10, 20, 65, 128),  // input_size
                                            testing::Values(1, 2, 3, 4),       // group
                                            testing::Values(2, 3, 4, 5),       // dim count
                                            testing::Values(DATA_TYPE_FLOAT, DATA_TYPE_HALF)));

TEST_P(GroupNormLayerTest, GroupNormLayer) {
    // get param
    int batch          = std::get<0>(GetParam());
    int channel        = std::get<1>(GetParam());
    int input_size     = std::get<2>(GetParam());
    int group          = std::get<3>(GetParam());
    int dim_count      = std::get<4>(GetParam());
    DataType data_type = std::get<5>(GetParam());
    DeviceType dev     = ConvertDeviceType(FLAGS_dt);

    if (channel % group) {
        GTEST_SKIP();
    }

    if (DEVICE_HUAWEI_NPU == dev) {
        GTEST_SKIP();
    }
    if (CheckDataTypeSkip(data_type)) {
        GTEST_SKIP();
    }
    if (DEVICE_OPENCL == dev && ((channel / group % 4 != 0) || channel % 4 != 0) || dim_count > 2) {
        GTEST_SKIP();
    }

    // param
    std::shared_ptr<GroupNormLayerParam> param(new GroupNormLayerParam());
    param->name  = "GroupNorm";
    param->group = group;

    // generate interpreter
    std::vector<int> input_dims = {batch, channel};
    while (input_dims.size() < dim_count) {
        input_dims.push_back(input_size);
    }

    //groupnorm has no layer resource, scale and bias weights are saved in constant map
    auto interpreter = GenerateInterpreter("GroupNorm", {input_dims, {channel}, {channel}}, param);
    if (DEVICE_APPLE_NPU == dev) {
        // resource
        int norm_size = channel;
        std::shared_ptr<RawBuffer> scales_buffer(new RawBuffer(sizeof(float) * norm_size));
        InitRandom(scales_buffer->force_to<float*>(), norm_size, 1.0f);
        std::shared_ptr<RawBuffer> biases_buffer(new RawBuffer(sizeof(float) * norm_size));
        InitRandom(biases_buffer->force_to<float*>(), norm_size, 1.0f);
        if (group % 2) {
            // bias may be empty
            biases_buffer = std::make_shared<RawBuffer>();
        }
        scales_buffer->SetBufferDims({channel});
        biases_buffer->SetBufferDims({channel});

        auto default_interpreter = dynamic_cast<DefaultModelInterpreter*>(interpreter.get());
        auto net_structure       = default_interpreter->GetNetStructure();
        auto net_resource        = default_interpreter->GetNetResource();
        net_resource->constant_map[net_structure->layers[0]->inputs[1]] = scales_buffer;
        net_resource->constant_map[net_structure->layers[0]->inputs[2]] = biases_buffer;
    }
    Precision precision = SetPrecision(dev, data_type);
    Run(interpreter, precision);
}

}  // namespace TNN_NS
