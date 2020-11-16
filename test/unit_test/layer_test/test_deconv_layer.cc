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
#include <fstream>
#include <streambuf>
#include <string>

#include "test/unit_test/layer_test/layer_test.h"
#include "test/unit_test/unit_test_common.h"
#include "test/unit_test/utils/network_helpers.h"
#include "tnn/interpreter/tnn/model_interpreter.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

class DeconvLayerTest
    : public LayerTest,
      public ::testing::WithParamInterface<std::tuple<int, int, int, int, int, int, int, int, int, int, DataType, int>> {};
INSTANTIATE_TEST_SUITE_P(LayerTest, DeconvLayerTest,
                         ::testing::Combine(testing::Values(1), testing::Values(1, 2, 3, 4, 13), testing::Values(1, 2, 3, 4, 16),
                                            // input_size
                                            testing::Values(2, 3, 8, 15),
                                            // group
                                            testing::Values(1, 2),
                                            // kernel
                                            testing::Values(1, 2, 3, 4),
                                            // dilation
                                            testing::Values(1),
                                            // stride
                                            testing::Values(2),
                                            // pads
                                            testing::Values(1),
                                            // output_pads
                                            testing::Values(0),
                                            // data_type
                                            testing::Values(DATA_TYPE_FLOAT, DATA_TYPE_BFP16),
                                            // activation_type
                                            testing::Values(ActivationType_None, ActivationType_ReLU, ActivationType_ReLU6, ActivationType_SIGMOID_MUL)));

TEST_P(DeconvLayerTest, DeconvLayer) {
    // get param
    int batch             = std::get<0>(GetParam());
    int input_channel_per_group = std::get<1>(GetParam());
    int output_channel_per_group = std::get<2>(GetParam());
    int input_size        = std::get<3>(GetParam());
    int group             = std::get<4>(GetParam());
    int kernel            = std::get<5>(GetParam());
    int dilation          = std::get<6>(GetParam());
    int stride            = std::get<7>(GetParam());
    int pad               = std::get<8>(GetParam());
    int output_pad        = std::get<9>(GetParam());
    auto dtype            = std::get<10>(GetParam());
    int activation_type   = std::get<11>(GetParam());

    DeviceType dev = ConvertDeviceType(FLAGS_dt);

    if (dtype == DATA_TYPE_BFP16 && DEVICE_ARM != dev) {
        GTEST_SKIP();
    }
    
    if ( DEVICE_METAL == dev && group != 1 && !(input_channel_per_group % 4 == 0 && output_channel_per_group % 4 == 0) && !(group == 2 && output_channel_per_group == 1 && input_channel_per_group == 2) ) {
        GTEST_SKIP();
    }

    if (kernel <= 1) {
        pad = 0;
    } else if (kernel == 2) {
        output_pad = 1;
    }

    int input_channel = group * input_channel_per_group;
    int output_channel = group * output_channel_per_group;
    // blob desc
    auto inputs_desc  = CreateInputBlobsDesc(batch, input_channel, input_size, 1, dtype);
    auto outputs_desc = CreateOutputBlobsDesc(1, dtype);

    // deconv param
    ConvLayerParam param;
    param.name           = "Deconv";
    param.input_channel  = input_channel;
    param.output_channel = output_channel;
    param.group          = group;
    param.kernels        = {kernel, kernel};
    param.dialations     = {dilation, dilation};
    param.strides        = {stride, stride};
    param.activation_type = activation_type;

    param.pads = {pad, pad, pad, pad};
    param.bias = 1;
    if (output_pad > 0) {
        param.pad_type = 3;
    }

    // resource
    ConvLayerResource resource;
    int filter_count = input_channel * output_channel * kernel * kernel / group;
    RawBuffer filter(filter_count * sizeof(float));
    float* filter_data = filter.force_to<float*>();
    RawBuffer bias(output_channel * sizeof(float));
    float* bias_data = bias.force_to<float*>();

    InitRandom(filter_data, filter_count, 1.0f);
    InitRandom(bias_data, output_channel, 1.0f);
    resource.filter_handle = filter;
    resource.bias_handle   = bias;

    Run(LAYER_DECONVOLUTION, &param, &resource, inputs_desc, outputs_desc);
}

}  // namespace TNN_NS
