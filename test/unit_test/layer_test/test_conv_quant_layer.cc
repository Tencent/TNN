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
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

class ConvQuantLayerTest : public LayerTest,
                           public ::testing::WithParamInterface<std::tuple<int, int, int, int, int, int, DataType>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, ConvQuantLayerTest,
                         ::testing::Combine(testing::Values(1), testing::Values(1, 2, 3, 4, 10, 32),
                                            testing::Values(9, 10, 16, 19),
                                            // kernel
                                            testing::Values(1, 3),
                                            // stride
                                            testing::Values(1, 2),
                                            // group
                                            testing::Values(1, 2, 3, 8),
                                            // data_type
                                            testing::Values(DATA_TYPE_INT8, DATA_TYPE_BFP16)));

TEST_P(ConvQuantLayerTest, ConvLayer) {
    // get param
    int batch             = std::get<0>(GetParam());
    int channel_per_group = std::get<1>(GetParam());
    int input_size        = std::get<2>(GetParam());
    int kernel            = std::get<3>(GetParam());
    int stride            = std::get<4>(GetParam());
    DataType data_type    = std::get<6>(GetParam());
    int group             = std::get<5>(GetParam());
    int channel           = group * channel_per_group;
    DeviceType dev        = ConvertDeviceType(FLAGS_dt);
    if (DEVICE_ARM != dev) {
        GTEST_SKIP();
    }

    // blob desc
    auto inputs_desc  = CreateInputBlobsDesc(batch, channel, input_size, 1, data_type);
    auto outputs_desc = CreateOutputBlobsDesc(1, data_type);

    // param
    ConvLayerParam param;
    param.name           = "Conv";
    param.input_channel  = channel;
    param.output_channel = channel;
    param.group          = group;
    param.kernels        = {kernel, kernel};
    param.dialations     = {1, 1};
    param.strides        = {stride, stride};
    param.pads           = {kernel / 2, kernel / 2, kernel / 2, kernel / 2};
    param.bias           = 1;

    // resource
    ConvLayerResource resource;
    auto element_size = (data_type == DATA_TYPE_INT8) ? 1 : 4;
    int filter_count  = channel * channel * kernel * kernel / group;
    RawBuffer filter(filter_count * element_size);
    RawBuffer bias(channel * sizeof(float));
    if (data_type == DATA_TYPE_BFP16) {
        InitRandom(filter.force_to<float*>(), filter_count, (float)1.0);
        InitRandom(bias.force_to<float*>(), channel, (float)1);
    } else {
        RawBuffer scale(channel * sizeof(float));
        InitRandom(filter.force_to<int8_t*>(), filter_count, (int8_t)8);
        filter.SetDataType(data_type);
        InitRandom(bias.force_to<int32_t*>(), channel, (int32_t)8);
        InitRandom(scale.force_to<float*>(), channel, 0.f, 1.0f);
        resource.scale_handle = scale;
    }

    resource.filter_handle = filter;
    resource.bias_handle   = bias;

    Run(LAYER_CONVOLUTION, &param, &resource, inputs_desc, outputs_desc);
}

}  // namespace TNN_NS
