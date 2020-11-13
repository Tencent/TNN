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

class PowLayerTest : public LayerTest,
                     public ::testing::WithParamInterface<std::tuple<int, int, int, float, float, float, DataType>> {};

INSTANTIATE_TEST_SUITE_P(LayerTest, PowLayerTest,
                         ::testing::Combine(
                             // batch
                             testing::Values(1),
                             // channel Values(1, 8),
                             testing::Values(1, 4, 15),
                             // size Values(16, 19),
                             testing::Values(1, 6, 8, 13),
                             // scale
                             testing::Values(1.234, 2.30, 0),
                             // shift
                             testing::Values(1.234, 1.234, 0.564),
                             // exponent
                             testing::Values(1.234, 2, 2.1),
                             // data_type
                             testing::Values(DATA_TYPE_FLOAT)));

TEST_P(PowLayerTest, PowLayer) {
    ensure_input_positive_ = 1;

    // get param
    int batch      = std::get<0>(GetParam());
    int channel    = std::get<1>(GetParam());
    int input_size = std::get<2>(GetParam());
    float scale    = std::get<3>(GetParam());
    float shift    = std::get<4>(GetParam());
    float exponent = std::get<5>(GetParam());

    DataType data_type = std::get<6>(GetParam());
    DeviceType dev     = ConvertDeviceType(FLAGS_dt);

    if (DEVICE_HUAWEI_NPU == dev) {
        GTEST_SKIP();
    }

    if (data_type == DATA_TYPE_INT8 && DEVICE_ARM != dev) {
        GTEST_SKIP();
    }

    // param
    std::shared_ptr<PowLayerParam> param(new PowLayerParam());
    param->name     = "Pow";
    param->scale    = scale;
    param->shift    = shift;
    param->exponent = exponent;

    // generate interpreter
    std::vector<int> input_dims = {batch, channel, input_size, input_size};
    auto interpreter            = GenerateInterpreter("Power", {input_dims}, param);
    Run(interpreter);
}

}  // namespace TNN_NS
