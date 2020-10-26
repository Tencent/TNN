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

class ConvLayerTest
    : public LayerTest,
      public ::testing::WithParamInterface<std::tuple<int, int, int, int, int, int, int, int, DataType>> {
    float GetCalcMflops(LayerParam* param, std::vector<Blob*> inputs, std::vector<Blob*> outputs) {
        ConvLayerParam* conv_param = dynamic_cast<ConvLayerParam*>(param);
        auto dims_input            = inputs[0]->GetBlobDesc().dims;
        auto dims_output           = outputs[0]->GetBlobDesc().dims;
        float Mflops = 2.0f * dims_output[0] * dims_output[1] * dims_input[1] * dims_output[2] * dims_output[3] *
                       conv_param->kernels[0] * conv_param->kernels[1] / 1000.f / 1000.f;
        return Mflops;
    }
};

INSTANTIATE_TEST_SUITE_P(LayerTest, ConvLayerTest,
                        ::testing::Combine(  // batch
                            testing::Values(1),
                            // channel
                            testing::Values(1, 2, 3, 4, 10, 32),
                            // hw
                            testing::Values(9, 10, 16, 19),
                            // group
                            testing::Values(1, 2),
                            // kernel
                            testing::Values(1, 2, 3, 5),
                            // dilation
                            testing::Values(1, 2),
                            // stride
                            testing::Values(1, 2),
                            // pads
                            testing::Values(0, 1),
                            // data_type
                            testing::Values(DATA_TYPE_FLOAT)));

TEST_P(ConvLayerTest, ConvLayer) {
    // get param
    int batch             = std::get<0>(GetParam());
    int channel_per_group = std::get<1>(GetParam());
    int input_size        = std::get<2>(GetParam());
    int group             = std::get<3>(GetParam());
    int channel           = group * channel_per_group;
    int kernel            = std::get<4>(GetParam());
    int dilation          = std::get<5>(GetParam());
    int stride            = std::get<6>(GetParam());
    int pad               = std::get<7>(GetParam());
    auto dtype            = std::get<8>(GetParam());
    DeviceType dev        = ConvertDeviceType(FLAGS_dt);

    if (dtype == DATA_TYPE_BFP16 && DEVICE_ARM != dev) {
        GTEST_SKIP();
    }

    if (((channel_per_group % 4) != 0) && DEVICE_METAL == dev) {
        GTEST_SKIP();
    }

    // blob desc
    auto inputs_desc  = CreateInputBlobsDesc(batch, channel, input_size, 1, DATA_TYPE_FLOAT);
    auto outputs_desc = CreateOutputBlobsDesc(1, DATA_TYPE_FLOAT);

    // param
    ConvLayerParam param;
    param.name            = "Conv";
    param.input_channel   = channel;
    param.output_channel  = channel;
    param.group           = group;
    param.kernels         = {kernel, kernel};
    param.dialations      = {dilation, dilation};
    param.strides         = {stride, stride};
    param.pads            = {pad, pad, pad, pad};
    param.bias            = 1;
    param.activation_type = ActivationType_ReLU;

    // resource
    ConvLayerResource resource;
    int filter_count = channel * channel * kernel * kernel / group;
    RawBuffer filter(filter_count * sizeof(float));
    float* filter_data = filter.force_to<float*>();
    RawBuffer bias(channel * sizeof(float));
    float* bias_data = bias.force_to<float*>();
    InitRandom(filter_data, filter_count, 1.0f);
    InitRandom(bias_data, channel, 1.0f);
    resource.filter_handle = filter;
    resource.bias_handle   = bias;

    Run(LAYER_CONVOLUTION, &param, &resource, inputs_desc, outputs_desc);
}

TEST_P(ConvLayerTest, ConvLayerWithProto) {
    // get param
    int batch             = std::get<0>(GetParam());
    int channel_per_group = std::get<1>(GetParam());
    int input_size        = std::get<2>(GetParam());
    int group             = std::get<3>(GetParam());
    int channel           = group * channel_per_group;
    int kernel            = std::get<4>(GetParam());
    int dilation          = std::get<5>(GetParam());
    int stride            = std::get<6>(GetParam());
    int pad               = std::get<7>(GetParam());
    auto dtype            = std::get<8>(GetParam());

    DeviceType dev        = ConvertDeviceType(FLAGS_dt);

    if (dtype == DATA_TYPE_BFP16 && DEVICE_ARM != dev) {
        GTEST_SKIP();
    }

    if (((channel_per_group % 4) != 0) && DEVICE_METAL == dev) {
        GTEST_SKIP();
    }

    // generate proto string
    std::string head = GenerateHeadProto({batch, channel, input_size, input_size});
    std::ostringstream ostr;
    ostr << "\"" << "Convolution conv 1 1 input output "
        << group << " " << channel_per_group << " " << channel << " "
        << kernel << " " << kernel << " " << stride << " " << stride << " "
        << pad << " " << pad << " 1 -1 " << dilation << " " << dilation << " "
        << ",\"";

    std::string proto = head + ostr.str();
    RunWithProto(proto);
}

}  // namespace TNN_NS
