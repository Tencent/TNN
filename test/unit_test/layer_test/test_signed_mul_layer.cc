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

namespace TNN_NS {

class SignedMulLayerTest
    : public LayerTest,
      public ::testing::WithParamInterface<std::tuple<int, int, int, float, float, float, DataType>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, SignedMulLayerTest,
                         ::testing::Combine(BASIC_BATCH_CHANNEL_SIZE,
                                            // alpha
                                            testing::Values(1.0f, 0.0f, -1.0f),
                                            // beta
                                            testing::Values(1.0f, 0.0f, -1.0f),
                                            // gema
                                            testing::Values(2.0f, -1.0f),
                                            // data type
                                            testing::Values(DATA_TYPE_FLOAT, DATA_TYPE_BFP16)));

TEST_P(SignedMulLayerTest, SignedMulLayer) {
    // get param
    int batch          = std::get<0>(GetParam());
    int channel        = std::get<1>(GetParam());
    int input_size     = std::get<2>(GetParam());
    float alpha        = std::get<3>(GetParam());
    float beta         = std::get<4>(GetParam());
    float gamma        = std::get<5>(GetParam());
    DataType data_type = std::get<6>(GetParam());
    DeviceType dev     = ConvertDeviceType(FLAGS_dt);
    if (data_type == DATA_TYPE_INT8 && DEVICE_ARM != dev) {
        GTEST_SKIP();
    }

    if (data_type == DATA_TYPE_BFP16 && DEVICE_ARM != dev) {
        GTEST_SKIP();
    }

    if (DEVICE_HUAWEI_NPU == dev) {
        GTEST_SKIP();
    }

    // param
    std::shared_ptr<SignedMulLayerParam> param(new SignedMulLayerParam());
    param->alpha = alpha;
    param->beta  = beta;
    param->gamma = gamma;

    // generate interpreter
    std::vector<int> input_dims = {batch, channel, input_size, input_size};
    auto interpreter            = GenerateInterpreter("SignedMul", {input_dims}, param);
    Precision precision         = PRECISION_AUTO;
    if (DATA_TYPE_BFP16 == data_type) {
        precision = PRECISION_LOW;
    }

    Run(interpreter, precision);
}

}  // namespace TNN_NS
