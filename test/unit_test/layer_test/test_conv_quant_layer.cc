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
                           public ::testing::WithParamInterface<std::tuple<int, int, int, int, int, int, int, DataType,
                                                                           ActivationType, FusionType>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, ConvQuantLayerTest,
                         ::testing::Combine(testing::Values(1), testing::Values(1, 2, 3, 4, 10, 32, 64),
                                            testing::Values(9, 10, 16, 19),
                                            // kernel
                                            testing::Values(1, 3),
                                            // stride
                                            testing::Values(1, 2),
                                            // group
                                            testing::Values(1, 2, 3, 8),
                                            // dilation
                                            testing::Values(1, 2),
                                            // data_type
                                            testing::Values(DATA_TYPE_INT8, DATA_TYPE_BFP16),
                                            // activation_type
                                            testing::Values(ActivationType_None, ActivationType_ReLU),
                                            // fusion_type
                                            testing::Values(FusionType_None, FusionType_Conv_Add_Activation,
                                                            FusionType_Conv_Activation_Add)));

TEST_P(ConvQuantLayerTest, ConvLayer) {
    // get param
    int batch             = std::get<0>(GetParam());
    int channel_per_group = std::get<1>(GetParam());
    int input_size        = std::get<2>(GetParam());
    int kernel            = std::get<3>(GetParam());
    int stride            = std::get<4>(GetParam());
    int group             = std::get<5>(GetParam());
    int dilation          = std::get<6>(GetParam());
    DataType data_type    = std::get<7>(GetParam());
    auto activation_type  = std::get<8>(GetParam());
    auto fusion_type      = std::get<9>(GetParam());
    int channel           = group * channel_per_group;
    DeviceType dev        = ConvertDeviceType(FLAGS_dt);
    if (DEVICE_ARM != dev) {
        GTEST_SKIP();
    }

    if (fusion_type != FusionType_None) {
        // only int8 data type support conv add fusion
        if (group != 1 || data_type != DATA_TYPE_INT8) {
            GTEST_SKIP();
        }
    }

    // param
    std::shared_ptr<ConvLayerParam> param(new ConvLayerParam());
    param->name            = "Conv";
    param->input_channel   = channel;
    param->output_channel  = channel;
    param->group           = group;
    param->kernels         = {kernel, kernel};
    param->dialations      = {dilation, dilation};
    param->strides         = {stride, stride};
    param->pads            = {kernel / 2, kernel / 2, kernel / 2, kernel / 2};
    param->bias            = 1;
    param->activation_type = activation_type;
    param->fusion_type     = fusion_type;

    std::vector<int> conv_input_dims = {batch, channel, input_size, input_size};
    std::vector<std::vector<int>> input_vec;
    input_vec.push_back(conv_input_dims);

    // get add input dim
    if (fusion_type != FusionType_None) {
        auto inputs_desc              = CreateInputBlobsDesc(batch, channel, input_size, 1, data_type);
        Blob conv_input_blob          = Blob(inputs_desc[0]);
        std::vector<Blob*> conv_input = {&conv_input_blob};

        BlobDesc conv_output_desc;
        conv_output_desc.data_type     = data_type;
        conv_output_desc.device_type   = DEVICE_NAIVE;
        Blob conv_output_blob          = Blob(conv_output_desc);
        std::vector<Blob*> conv_output = {&conv_output_blob};

        auto layer_creator_map = GetGlobalLayerCreatorMap();
        auto conv_layer        = layer_creator_map[LAYER_CONVOLUTION]->CreateLayer();

        conv_layer->InferShapeAhead(conv_input, conv_output, param.get(), nullptr);
        input_vec.push_back(conv_output[0]->GetBlobDesc().dims);
        delete conv_layer;
    }

    // generate proto string
    Precision precision = PRECISION_AUTO;
    if (DATA_TYPE_INT8 == data_type) {
        param->quantized = true;
    } else if (DATA_TYPE_BFP16 == data_type) {
        precision = PRECISION_LOW;
    }

    auto interpreter = GenerateInterpreter("Convolution", input_vec, param);
    Run(interpreter, precision);
}

}  // namespace TNN_NS
