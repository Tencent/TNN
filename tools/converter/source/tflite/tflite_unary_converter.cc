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

#include "tflite_op_converter.h"
#include "tflite_utils.h"

namespace TNN_CONVERTER {

DECLARE_OP_CONVERTER(Unary);
std::string TFLiteUnaryConverter::TNNOpType(tflite::BuiltinOperator op_code, bool quantized_model) {
    switch (op_code) {
        case tflite::BuiltinOperator_LOGISTIC:
            return "Sigmoid";
        case tflite::BuiltinOperator_EXP:
            return "Exp";
        case tflite::BuiltinOperator_LOG:
            return "Log";
        case tflite::BuiltinOperator_TANH:
            return "Tanh";
        case tflite::BuiltinOperator_COS:
            return "Cos";
        case tflite::BuiltinOperator_SIN:
            return "Sin";
        case tflite::BuiltinOperator_NEG:
            return "Neg";
        case tflite::BuiltinOperator_RSQRT:
            return "Rsqrt";
        case tflite::BuiltinOperator_RELU:
            return "ReLU";
        case tflite::BuiltinOperator_SHAPE:
            return "Shape";
        case tflite::BuiltinOperator_SQRT:
            return "Sqrt";
        default:
            return "";
    }
}
tflite::ActivationFunctionType TFLiteUnaryConverter::ActivationType(
    const std::unique_ptr<tflite::OperatorT> &tf_lite_operator, tflite::BuiltinOperator op_code) {
    return tflite::ActivationFunctionType_NONE;
}

TNN_NS::Status TFLiteUnaryConverter::exec(TNN_NS::NetStructure &net_structure, TNN_NS::NetResource &net_resource,
                                          const std::unique_ptr<tflite::OperatorT> &tf_lite_operator,
                                          const std::vector<std::unique_ptr<tflite::TensorT>> &tf_lite_tensors,
                                          const std::vector<std::unique_ptr<tflite::BufferT>> &tf_lite_model_buffer,
                                          const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tf_lite_op_set,
                                          bool quantized_model) {
    auto param       = new TNN_NS::LayerParam;
    auto cur_layer   = net_structure.layers.back();
    cur_layer->param = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->type      = cur_layer->type_str;
    param->name      = cur_layer->name;
    param->quantized = false;
    return TNN_NS::TNN_CONVERT_OK;
}

using namespace tflite;
REGISTER_CONVERTER(Unary, BuiltinOperator_LOGISTIC);
REGISTER_CONVERTER(Unary, BuiltinOperator_EXP);
REGISTER_CONVERTER(Unary, BuiltinOperator_LOG);
REGISTER_CONVERTER(Unary, BuiltinOperator_TANH);
REGISTER_CONVERTER(Unary, BuiltinOperator_COS);
REGISTER_CONVERTER(Unary, BuiltinOperator_SIN);
REGISTER_CONVERTER(Unary, BuiltinOperator_NEG);
REGISTER_CONVERTER(Unary, BuiltinOperator_RSQRT);
REGISTER_CONVERTER(Unary, BuiltinOperator_RELU);
REGISTER_CONVERTER(Unary, BuiltinOperator_SHAPE);
REGISTER_CONVERTER(Unary, BuiltinOperator_SQRT);
}  // namespace TNN_CONVERTER
