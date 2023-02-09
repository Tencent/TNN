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
DECLARE_OP_CONVERTER(Pack);

std::string TFLitePackConverter::TNNOpType(tflite::BuiltinOperator op_code, bool quantized_model) {
    return "Concat";
}

tflite::ActivationFunctionType TFLitePackConverter::ActivationType(
    const std::unique_ptr<tflite::OperatorT> &tf_lite_operator, tflite::BuiltinOperator op_code) {
    return tflite::ActivationFunctionType_NONE;
}

TNN_NS::Status TFLitePackConverter::exec(TNN_NS::NetStructure &net_structure, TNN_NS::NetResource &net_resource,
                                         const std::unique_ptr<tflite::OperatorT> &tf_lite_operator,
                                         const std::vector<std::unique_ptr<tflite::TensorT>> &tf_lite_tensors,
                                         const std::vector<std::unique_ptr<tflite::BufferT>> &tf_lite_model_buffer,
                                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tf_lite_op_set,
                                         bool quantized_model) {
    auto param         = new TNN_NS::ConcatLayerParam;
    auto cur_layer     = net_structure.layers.back();
    cur_layer->param   = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->name        = cur_layer->name;
    param->type        = cur_layer->type_str;
    param->quantized   = quantized_model;
    auto option        = tf_lite_operator->builtin_options.AsPackOptions();
    auto &input_tensor = tf_lite_tensors[tf_lite_operator->inputs[0]];
    param->axis        = ConvertAxisFormatTFLite(option->axis, input_tensor->shape.size());
    for (const auto &index : tf_lite_operator->inputs) {
        const auto &tensor = tf_lite_tensors[index];
        const auto &buffer = tf_lite_model_buffer[tensor->buffer];
        if (buffer->data.empty()) {
            continue;
        }
        const auto &input_dims = tensor->shape;
        int data_count         = buffer->data.size() / SizeofTFLiteTensorData(tensor->type);
        if (input_dims.empty() && data_count == 1 && tensor->type == tflite::TensorType_INT32) {
            // only one value
            std::vector<int32_t> tnn_dims = {1};
            auto raw_buffer               = new TNN_NS::RawBuffer(data_count * sizeof(int32_t), tnn_dims);
            raw_buffer->SetDataType(TNN_NS::DATA_TYPE_INT32);
            ::memcpy(raw_buffer->force_to<int32_t *>(), reinterpret_cast<int32_t *>(buffer->data.data()),
                     data_count * sizeof(int32_t));
            net_resource.constant_map[tensor->name] = std::shared_ptr<TNN_NS::RawBuffer>(raw_buffer);
        } else {
            LOGE("TFLite Pack only support pack one value");
            return TNN_NS::Status(TNN_NS::TNNERR_UNSUPPORT_NET, "TFLite Pack only support pack one value");
        }
    }
    return TNN_NS::TNN_CONVERT_OK;
}

using namespace tflite;
REGISTER_CONVERTER(Pack, BuiltinOperator_PACK);

}  // namespace TNN_CONVERTER
