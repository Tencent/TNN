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
#include "tnn/utils/cpu_utils.h"

namespace TNN_NS {

UnaryLayerTest::UnaryLayerTest(LayerType type) {
    layer_type_ = type;
}

void UnaryLayerTest::RunUnaryTest(std::string type_str) {
    // get param
    int batch          = std::get<0>(GetParam());
    int channel        = std::get<1>(GetParam());
    int input_size     = std::get<2>(GetParam());
    int dim_count      = std::get<3>(GetParam());
    DataType data_type = std::get<4>(GetParam());
    DeviceType dev     = ConvertDeviceType(FLAGS_dt);

    if(CheckDataTypeSkip(data_type)) {
        GTEST_SKIP();
    }

    //special for cuda skip
    if ((type_str == "Reciprocal" || type_str == "Softplus") && DEVICE_CUDA == dev) {
        GTEST_SKIP();
    }

        //special for apple npu skip
    if ((type_str == "Reciprocal" || type_str == "Abs" || type_str == "Acos" || type_str == "Sin"|| type_str == "Log" ||
         type_str == "Asin" || type_str == "Atan"|| type_str == "Ceil" || type_str == "Cos" || type_str == "Floor" ||
         type_str == "LogSigmoid" || type_str == "Neg" || type_str == "Tan" || type_str == "Sqrt" || type_str == "Sign")
        && DEVICE_APPLE_NPU == dev) {
        GTEST_SKIP();
    }

    // skip dims > 4 for HUAWEI_NPU
    if (dim_count > 4 && DEVICE_HUAWEI_NPU == dev) {
        GTEST_SKIP();
    }

    std::shared_ptr<LayerParam> param(new LayerParam());
    param->name = "Unary";

    Precision precision = SetPrecision(dev, data_type);
    
    // generate proto string
    std::vector<int> input_dims = {batch, channel};
    while(input_dims.size() < dim_count) input_dims.push_back(input_size);
    if (DATA_TYPE_INT8 == data_type) {
        param->quantized = true;
    }

    auto interpreter = GenerateInterpreter(type_str, {input_dims}, param);
    Run(interpreter, precision);
}

}  // namespace TNN_NS
