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
#include "tnn/utils/data_format_converter.h"

namespace TNN_CONVERTER {
DECLARE_OP_CONVERTER(Binary);

std::string TFLiteBinaryConverter::TNNOpType(tflite::BuiltinOperator op_code, bool quantized_model) {
    std::string prefix;
    if (quantized_model) {
        prefix = "Quantized";
    }
    switch (op_code) {
        case tflite::BuiltinOperator_ADD:
            return prefix + "Add";
        case tflite::BuiltinOperator_SUB:
            return prefix + "Sub";
        case tflite::BuiltinOperator_MUL:
            return prefix + "Mul";
        case tflite::BuiltinOperator_DIV:
            return prefix + "Div";
        case tflite::BuiltinOperator_MAXIMUM:
            return prefix + "Maximum";
        case tflite::BuiltinOperator_MINIMUM:
            return prefix + "Minimum";
        case tflite::BuiltinOperator_SQUARED_DIFFERENCE:
            return prefix + "SquaredDifference";
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
    param->quantized          = quantized_model;
    param->weight_input_index = -1;
    for (int i = 0; i < tf_lite_operator->inputs.size(); ++i) {
        auto& tensor = tf_lite_tensors[tf_lite_operator->inputs[i]];
        auto& buffer = tf_lite_model_buffer[tensor->buffer];
        if (!buffer->data.empty()) {
            assert(param->weight_input_index == -1);
            param->weight_input_index = i;
        }
    }
    if (param->weight_input_index != -1) {
        // get weight from weight
        auto layer_resource     = new TNN_NS::EltwiseLayerResource;
        layer_resource->name    = cur_layer->name;
        auto& weight_tensor     = tf_lite_tensors[tf_lite_operator->inputs[param->weight_input_index]];
        const auto& weight_dims = weight_tensor->shape;
        auto weight_size =
            tf_lite_model_buffer[weight_tensor->buffer]->data.size() / SizeofTFLiteTensorData(weight_tensor->type);
        const auto& input_index  = param->weight_input_index == 0 ? 1 : 0;
        const auto& input_tensor = tf_lite_tensors[tf_lite_operator->inputs[input_index]];
        const auto& input_dims   = input_tensor->shape;
        auto tnn_dims            = std::vector<int32_t>(input_dims.size(), 1);
        if (weight_size != 0) {
            if (weight_dims.empty() || weight_dims.size() == 1) {
                if (input_dims.size() == 1) {
                    tnn_dims[0] = weight_size;
                } else {
                    tnn_dims[1] = weight_size;
                }
            } else {
                tnn_dims = weight_dims;
                ConvertShapeFormatTFLite(tnn_dims);
            }
        } else {
            return TNN_NS::Status(TNN_NS::TNNERR_CONVERT_INVALID_MODEL, "TFLite:The weight size should not be zero!\n");
        }
        auto weight_ptr = reinterpret_cast<float*>(tf_lite_model_buffer[weight_tensor->buffer]->data.data());
        TNN_NS::RawBuffer element_handle = TNN_NS::RawBuffer(weight_size * sizeof(float), tnn_dims);
        if (tnn_dims.size() < 3) {
            ::memcpy(element_handle.force_to<float*>(), weight_ptr, weight_size * sizeof(float));
        } else if (tnn_dims.size() == 3) {
            const auto& n = tnn_dims[0];
            const auto& c = tnn_dims[1];
            const auto& h = tnn_dims[2];
            const auto& w = 1;
            TNN_NS::DataFormatConverter::ConvertBetweenNHWCAndNCHW<float>(
                weight_ptr, element_handle.force_to<float*>(), n, c, h, w, TNN_NS::DataFormatConverter::NHWC2NCHW);
        } else if (tnn_dims.size() == 4) {
            const auto& n = tnn_dims[0];
            const auto& c = tnn_dims[1];
            const auto& h = tnn_dims[2];
            const auto& w = tnn_dims[3];
            TNN_NS::DataFormatConverter::ConvertBetweenNHWCAndNCHW<float>(
                weight_ptr, element_handle.force_to<float*>(), n, c, h, w, TNN_NS::DataFormatConverter::NHWC2NCHW);
        }
        layer_resource->element_handle             = ConvertRawBuffer::GetInstance()->Convert(element_handle);
        net_resource.resource_map[cur_layer->name] = std::shared_ptr<TNN_NS::LayerResource>(layer_resource);
        cur_layer->inputs.resize(1);
        if (param->weight_input_index == 0) {
            cur_layer->inputs[0] = tf_lite_tensors[tf_lite_operator->inputs[1]]->name;
        } else {
            cur_layer->inputs[0] = tf_lite_tensors[tf_lite_operator->inputs[0]]->name;
        }
    } else {
        if (quantized_model) {
            // handle input
            for (const auto& index : tf_lite_operator->inputs) {
                auto status = CreateBlobScaleResource(net_resource, tf_lite_tensors, index);
                ASSERT(status == TNN_NS::TNN_CONVERT_OK);
            }
            // handle output
            const auto& output_index = tf_lite_operator->outputs[0];
            auto status              = CreateBlobScaleResource(net_resource, tf_lite_tensors, output_index);
            ASSERT(status);
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
