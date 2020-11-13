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

#include "test/unit_test/layer_test/test_unary_layer.h"

namespace TNN_NS {

UnaryLayerTest::UnaryLayerTest(LayerType type) {
    layer_type_ = type;
}

void UnaryLayerTest::RunUnaryTest(std::string type_str) {
    // get param
    int batch          = std::get<0>(GetParam());
    int channel        = std::get<1>(GetParam());
    int input_size     = std::get<2>(GetParam());
    DataType data_type = std::get<3>(GetParam());
    DeviceType dev     = ConvertDeviceType(FLAGS_dt);

    if (data_type == DATA_TYPE_INT8 && DEVICE_ARM != dev) {
        GTEST_SKIP();
    }
    if (data_type == DATA_TYPE_BFP16 && DEVICE_ARM != dev) {
        GTEST_SKIP();
    }

    std::shared_ptr<LayerParam> param(new LayerParam());
    param->name = "Unary";

    Precision precision = PRECISION_AUTO;
    // generate proto string
    std::vector<int> input_dims = {batch, channel, input_size, input_size};
    if (DATA_TYPE_INT8 == data_type) {
        param->quantized = true;
    } else if (DATA_TYPE_BFP16 == data_type) {
        precision = PRECISION_LOW;
    }

    auto interpreter = GenerateInterpreter(type_str, {input_dims}, param);
    Run(interpreter, precision);
}

}  // namespace TNN_NS
