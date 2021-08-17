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

DECLARE_OP_CONVERTER(ArgMaxOrMin);

std::string TFLiteArgMaxOrMinConverter::TNNOpType(tflite::BuiltinOperator op_code, bool quantized_model) {
    return "ArgMaxOrMin";
}

tflite::ActivationFunctionType TFLiteArgMaxOrMinConverter::ActivationType(
    const std::unique_ptr<tflite::OperatorT> &tf_lite_operator, tflite::BuiltinOperator op_code) {
    return tflite::ActivationFunctionType_NONE;
}

TNN_NS::Status TFLiteArgMaxOrMinConverter::exec(
    TNN_NS::NetStructure &net_structure, TNN_NS::NetResource &net_resource,
    const std::unique_ptr<tflite::OperatorT> &tf_lite_operator,
    const std::vector<std::unique_ptr<tflite::TensorT>> &tf_lite_tensors,
    const std::vector<std::unique_ptr<tflite::BufferT>> &tf_lite_model_buffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tf_lite_op_set, bool quantized_model) {
    auto param       = new TNN_NS::ArgMaxOrMinLayerParam;
    auto cur_layer   = net_structure.layers.back();
    cur_layer->param = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->name      = cur_layer->name;
    param->type      = cur_layer->type_str;
    param->quantized = false;
    // tflite always keep_dims = 0
    param->keep_dims    = 0;
    auto tflite_op_type = tf_lite_op_set[tf_lite_operator->opcode_index]->builtin_code;
    if (tflite_op_type == tflite::BuiltinOperator_ARG_MIN) {
        param->mode = 0;
    } else {
        param->mode = 1;
    }
    if (tf_lite_operator->inputs.size() != 2) {
        LOGE("TFLiteArgMaxOrMinConverter only support input size is 2, but now input size is %lu\n",
             tf_lite_operator->inputs.size());
        return TNN_NS::TNNERR_UNSUPPORT_NET;
    }
    auto &axes_tensor = tf_lite_tensors[tf_lite_operator->inputs[1]];
    ASSERT(axes_tensor->type == tflite::TensorType_INT32 || axes_tensor->type == tflite::TensorType_INT64);
    const auto &buffer = tf_lite_model_buffer[axes_tensor->buffer];
    int data_len       = buffer->data.size() / SizeofTFLiteTensorData(axes_tensor->type);
    ASSERT(data_len == 1);
    int axis = 0;
    if (axes_tensor->type == tflite::TensorType_INT32) {
        axis = reinterpret_cast<int32_t *>(buffer->data.data())[0];
    } else {
        axis = reinterpret_cast<int64_t *>(buffer->data.data())[0];
    }
    auto &input_tensor = tf_lite_tensors[tf_lite_operator->inputs[0]];
    param->axis        = ConvertAxisFormatTFLite(axis, input_tensor->shape.size());
    cur_layer->inputs.resize(1);
    cur_layer->inputs[0] = tf_lite_tensors[tf_lite_operator->inputs[0]]->name;
    return TNN_NS::TNN_CONVERT_OK;
}

using namespace tflite;
REGISTER_CONVERTER(ArgMaxOrMin, BuiltinOperator_ARG_MAX);
REGISTER_CONVERTER(ArgMaxOrMin, BuiltinOperator_ARG_MIN);

}  // namespace TNN_CONVERTER
