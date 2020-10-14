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
DECLARE_OP_CONVERTER(Binary);

std::string TFLiteBinaryConverter::TNNOpType(tflite::BuiltinOperator op_code, bool quantized_model) {
    switch (op_code) {
        case tflite::BuiltinOperator_ADD:
            return "Add";
        case tflite::BuiltinOperator_SUB:
            return "Sub";
        case tflite::BuiltinOperator_MUL:
            return "Mul";
        case tflite::BuiltinOperator_DIV:
            return "Div";
        case tflite::BuiltinOperator_MAXIMUM:
            return "Maximum";
        case tflite::BuiltinOperator_MINIMUM:
            return "Minimum";
        case tflite::BuiltinOperator_SQUARED_DIFFERENCE:
            return "SquaredDifference";
        default:
            return "";
    }
}

tflite::ActivationFunctionType TFLiteBinaryConverter::ActivationType(
    const std::unique_ptr<tflite::OperatorT>& tf_lite_operator, tflite::BuiltinOperator op_code) {
    switch (op_code) {
        case tflite::BuiltinOperator_ADD:
            return tf_lite_operator->builtin_options.AsAddOptions()->fused_activation_function;
        case tflite::BuiltinOperator_SUB:
            return tf_lite_operator->builtin_options.AsSubOptions()->fused_activation_function;
        case tflite::BuiltinOperator_MUL:
            return tf_lite_operator->builtin_options.AsMulOptions()->fused_activation_function;
        case tflite::BuiltinOperator_DIV:
            return tf_lite_operator->builtin_options.AsDivOptions()->fused_activation_function;
        case tflite::BuiltinOperator_MAXIMUM:
        case tflite::BuiltinOperator_MINIMUM:
        case tflite::BuiltinOperator_SQUARED_DIFFERENCE:
            return tflite::ActivationFunctionType_NONE;
        default:
            return tflite::ActivationFunctionType_NONE;
    }
}

TNN_NS::Status TFLiteBinaryConverter::exec(TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource,
                                           const std::unique_ptr<tflite::OperatorT>& tf_lite_operator,
                                           const std::vector<std::unique_ptr<tflite::TensorT>>& tf_lite_tensors,
                                           const std::vector<std::unique_ptr<tflite::BufferT>>& tf_lite_model_buffer,
                                           const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tf_lite_op_set,
                                           bool quantized_model) {
    auto param                = new TNN_NS::MultidirBroadcastLayerParam;
    auto cur_layer            = net_structure.layers.back();
    cur_layer->param          = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->type               = cur_layer->type_str;
    param->name               = cur_layer->name;
    param->quantized          = false;
    param->weight_input_index = -1;
    for (int i = 0; i < tf_lite_operator->inputs.size(); ++i) {
        auto& tensor = tf_lite_tensors[tf_lite_operator->inputs[i]];
        auto& buffer = tf_lite_model_buffer[tensor->buffer];
        if (!buffer->data.empty()) {
            assert(param->weight_input_index == -1);
            param->weight_input_index = i;
        }
    }
    if (quantized_model) {
        // TODO
    } else {
        if (param->weight_input_index != -1) {
            // get weight from weight
            auto layer_resource  = new TNN_NS::EltwiseLayerResource;
            layer_resource->name = cur_layer->name;
            auto& weight_tensor  = tf_lite_tensors[tf_lite_operator->inputs[param->weight_input_index]];
            auto weight_ptr      = reinterpret_cast<float*>(tf_lite_model_buffer[weight_tensor->buffer]->data.data());
            auto weight_size =
                tf_lite_model_buffer[weight_tensor->buffer]->data.size() / SizeofTFLiteTensorData(weight_tensor->type);
            TNN_NS::RawBuffer element_handle = TNN_NS::RawBuffer(weight_size * sizeof(float));
            ::memcpy(element_handle.force_to<float*>(), weight_ptr, weight_size * sizeof(float));
            layer_resource->element_handle             = element_handle;
            net_resource.resource_map[cur_layer->name] = std::shared_ptr<TNN_NS::LayerResource>(layer_resource);
            cur_layer->inputs.resize(1);
            if (param->weight_input_index == 0) {
                cur_layer->inputs[0] = tf_lite_tensors[tf_lite_operator->inputs[1]]->name;
            } else {
                cur_layer->inputs[0] = tf_lite_tensors[tf_lite_operator->inputs[0]]->name;
            }
        }
    }
    return TNN_NS::TNN_CONVERT_OK;
}

using namespace tflite;
REGISTER_CONVERTER(Binary, BuiltinOperator_ADD);
REGISTER_CONVERTER(Binary, BuiltinOperator_SUB);
REGISTER_CONVERTER(Binary, BuiltinOperator_MUL);
REGISTER_CONVERTER(Binary, BuiltinOperator_DIV);
REGISTER_CONVERTER(Binary, BuiltinOperator_MAXIMUM);
REGISTER_CONVERTER(Binary, BuiltinOperator_MINIMUM);
REGISTER_CONVERTER(Binary, BuiltinOperator_SQUARED_DIFFERENCE);

}  // namespace TNN_CONVERTER
