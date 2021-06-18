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

DECLARE_OP_CONVERTER(Square);

std::string TFLiteSquareConverter::TNNOpType(tflite::BuiltinOperator op_code, bool quantized_model) {
    return "Power";
}

tflite::ActivationFunctionType TFLiteSquareConverter::ActivationType(
    const std::unique_ptr<tflite::OperatorT>& tf_lite_operator, tflite::BuiltinOperator op_code) {
    return tflite::ActivationFunctionType_NONE;
}

TNN_NS::Status TFLiteSquareConverter::exec(TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource,
                                           const std::unique_ptr<tflite::OperatorT>& tf_lite_operator,
                                           const std::vector<std::unique_ptr<tflite::TensorT>>& tf_lite_tensors,
                                           const std::vector<std::unique_ptr<tflite::BufferT>>& tf_lite_model_buffer,
                                           const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tf_lite_op_set,
                                           bool quantized_model) {
    auto* param                = new TNN_NS::PowLayerParam;
    auto cur_layer             = net_structure.layers.back();
    cur_layer->param           = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->name                = cur_layer->name;
    param->type                = cur_layer->type_str;
    param->quantized           = false;
    param->scale               = 1.0;
    param->shift               = 0.0;
    const auto tf_lite_op_type = tf_lite_op_set[tf_lite_operator->opcode_index]->builtin_code;
    if (tf_lite_op_type == tflite::BuiltinOperator_SQRT) {
        param->exponent = 2.0;
    } else if (tf_lite_op_type == tflite::BuiltinOperator_POW) {
        ASSERT(tf_lite_operator->inputs.size() == 2);
        const auto& tensor = tf_lite_tensors[tf_lite_operator->inputs[1]];
        const auto& buffer = tf_lite_model_buffer[tensor->buffer];
        const auto count   = buffer->data.size() / SizeofTFLiteTensorData(tensor->type);
        ASSERT(count == 1);
        param->exponent = reinterpret_cast<float*>(buffer->data.data())[0];
        cur_layer->inputs.resize(1);
        cur_layer->inputs[0] = tf_lite_tensors[tf_lite_operator->inputs[0]]->name;
    } else {
        LOGE("TFLite Convert: do not support operator type");
        return TNN_NS::TNNERR_MODEL_ERR;
    }

    return TNN_NS::TNN_CONVERT_OK;
}

using namespace tflite;
REGISTER_CONVERTER(Square, BuiltinOperator_SQUARE);
REGISTER_CONVERTER(Square, BuiltinOperator_POW);
}  // namespace TNN_CONVERTER
