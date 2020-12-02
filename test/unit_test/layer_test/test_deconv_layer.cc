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

class DeconvLayerTest : public LayerTest,
                        public ::testing::WithParamInterface<
                            std::tuple<int, int, int, int, int, int, int, int, int, int, DataType, int>> {};
INSTANTIATE_TEST_SUITE_P(LayerTest, DeconvLayerTest,
                         ::testing::Combine(testing::Values(1), testing::Values(1, 2, 3, 4, 13),
                                            testing::Values(1, 2, 3, 4, 16),
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
                                            testing::Values(ActivationType_None, ActivationType_ReLU,
                                                            ActivationType_ReLU6, ActivationType_SIGMOID_MUL)));

TEST_P(DeconvLayerTest, DeconvLayer) {
    // get param
    int batch                    = std::get<0>(GetParam());
    int input_channel_per_group  = std::get<1>(GetParam());
    int output_channel_per_group = std::get<2>(GetParam());
    int input_size               = std::get<3>(GetParam());
    int group                    = std::get<4>(GetParam());
    int kernel                   = std::get<5>(GetParam());
    int dilation                 = std::get<6>(GetParam());
    int stride                   = std::get<7>(GetParam());
    int pad                      = std::get<8>(GetParam());
    int output_pad               = std::get<9>(GetParam());
    auto data_type               = std::get<10>(GetParam());
    int activation_type          = std::get<11>(GetParam());

    DeviceType dev = ConvertDeviceType(FLAGS_dt);

    if (data_type == DATA_TYPE_BFP16 && DEVICE_ARM != dev) {
        GTEST_SKIP();
    }

    if (DEVICE_METAL == dev && group != 1 && !(input_channel_per_group % 4 == 0 && output_channel_per_group % 4 == 0) &&
        !(group == 2 && output_channel_per_group == 1 && input_channel_per_group == 2)) {
        GTEST_SKIP();
    }

    if (DEVICE_HUAWEI_NPU == dev && activation_type != ActivationType_None) {
        GTEST_SKIP();
    }

    if (kernel <= 1) {
        pad = 0;
    } else if (kernel == 2) {
        output_pad = 1;
    }

    int input_channel  = group * input_channel_per_group;
    int output_channel = group * output_channel_per_group;

    // deconv param
    std::shared_ptr<ConvLayerParam> param(new ConvLayerParam());
    param->name            = "Deconv";
    param->input_channel   = input_channel;
    param->output_channel  = output_channel;
    param->group           = group;
    param->kernels         = {kernel, kernel};
    param->dialations      = {dilation, dilation};
    param->strides         = {stride, stride};
    param->activation_type = activation_type;

    param->pads = {pad, pad, pad, pad};
    param->bias = 1;

    if (DEVICE_HUAWEI_NPU == dev) {
        param->bias = 0;
    }

    if (output_pad > 0) {
        param->pad_type = 3;
    }

    if (param->pad_type != 0 && param->pad_type != 1 && param->pad_type != -1 && DEVICE_HUAWEI_NPU == dev) {
        GTEST_SKIP();
    }

    Precision precision = PRECISION_AUTO;
    if (DATA_TYPE_BFP16 == data_type) {
        precision = PRECISION_LOW;
    }

    if (DEVICE_ARM == dev && ActivationType_SIGMOID_MUL) {
        if (DATA_TYPE_FLOAT == data_type) {
            precision = PRECISION_HIGH;
        } else {
            GTEST_SKIP();
        }
    }

    // generate interpreter
    std::vector<int> input_dims = {batch, input_channel, input_size, input_size};
    auto interpreter            = GenerateInterpreter("Deconvolution", {input_dims}, param);
    Run(interpreter, precision);
}

}  // namespace TNN_NS
