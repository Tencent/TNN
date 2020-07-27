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

#include "tflite_binary_converter.h"

namespace TNN_CONVERTER {

TNN_NS::Status TFLiteBinaryConverter::exec(TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource,
                                           const std::unique_ptr<tflite::OperatorT>& tf_lite_operator,
                                           const std::vector<std::unique_ptr<tflite::TensorT>>& tf_lite_tensors,
                                           const std::vector<std::unique_ptr<tflite::BufferT>>& tf_lite_model_buffer,
                                           const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tf_lite_op_set,
                                           bool quantized_model) {
    auto param     = new TNN_NS::MultidirBroadcastLayerParam;
    auto cur_layer = net_structure.layers.back();
    cur_layer->param = std::shared_ptr<TNN_NS::LayerParam>(param);

    // TODO support resource

    return TNN_NS::TNN_CONVERT_OK;
}

DECLARE_BINARY_CONVERTER(Add);
std::string TFLiteAddConverter::TNNOpType(bool quantized_model) {
    return "Add";
};
DECLARE_BINARY_CONVERTER(Sub);
std::string TFLiteSubConverter::TNNOpType(bool quantized_model) {
    return "Sub";
};
DECLARE_BINARY_CONVERTER(Mul);
std::string TFLiteMulConverter::TNNOpType(bool quantized_model) {
    return "Mul";
};
DECLARE_BINARY_CONVERTER(Div);
std::string TFLiteDivConverter::TNNOpType(bool quantized_model) {
    return "Div";
};

using namespace tflite;
REGISTER_CONVERTER(Add, BuiltinOperator_ADD);
REGISTER_CONVERTER(Sub, BuiltinOperator_SUB);
REGISTER_CONVERTER(Mul, BuiltinOperator_MUL);
REGISTER_CONVERTER(Div, BuiltinOperator_DIV);

}  // namespace TNN_CONVERTER
