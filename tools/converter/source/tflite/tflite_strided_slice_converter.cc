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
DECLARE_OP_CONVERTER(StridedSlice);

std::string TFLiteStridedSliceConverter::TNNOpType(tflite::BuiltinOperator op_code, bool quantized_model) {
    return "StridedSlice";
}

tflite::ActivationFunctionType TFLiteStridedSliceConverter::ActivationType(
    const std::unique_ptr<tflite::OperatorT> &tf_lite_operator, tflite::BuiltinOperator op_code) {
    return tflite::ActivationFunctionType_NONE;
}

TNN_NS::Status TFLiteStridedSliceConverter::exec(
    TNN_NS::NetStructure &net_structure, TNN_NS::NetResource &net_resource,
    const std::unique_ptr<tflite::OperatorT> &tf_lite_operator,
    const std::vector<std::unique_ptr<tflite::TensorT>> &tf_lite_tensors,
    const std::vector<std::unique_ptr<tflite::BufferT>> &tf_lite_model_buffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tf_lite_op_set, bool quantized_model) {
    auto parm        = new TNN_NS::StrideSliceLayerParam;
    auto cur_layer   = net_structure.layers.back();
    cur_layer->param = std::shared_ptr<TNN_NS::LayerParam>(parm);
    parm->type       = cur_layer->type_str;
    parm->name       = cur_layer->name;
    parm->quantized  = false;
    auto option      = tf_lite_operator->builtin_options.AsStridedSliceOptions();
    ASSERT(tf_lite_operator->inputs.size() >= 3);
    auto &input_tensor = tf_lite_tensors[tf_lite_operator->inputs[0]];
    auto &begin_tensor = tf_lite_tensors[tf_lite_operator->inputs[1]];
    auto begin_size =
        tf_lite_model_buffer[begin_tensor->buffer]->data.size() / SizeofTFLiteTensorData(begin_tensor->type);
    auto begin_data_ptr = reinterpret_cast<int32_t *>(tf_lite_model_buffer[begin_tensor->buffer]->data.data());
    for (int i = 0; i < begin_size; ++i) {
        parm->begins.push_back(begin_data_ptr[i]);
    }
    Mask(input_tensor->shape, option->begin_mask, 0, parm->begins);
    ConvertShapeFormatTFLite(parm->begins);
    std::reverse(parm->begins.begin(), parm->begins.end());

    auto &end_tensor = tf_lite_tensors[tf_lite_operator->inputs[2]];
    auto end_size    = tf_lite_model_buffer[end_tensor->buffer]->data.size() / SizeofTFLiteTensorData(end_tensor->type);
    auto end_data_ptr = reinterpret_cast<int32_t *>(tf_lite_model_buffer[end_tensor->buffer]->data.data());
    for (int i = 0; i < end_size; ++i) {
        parm->ends.push_back(end_data_ptr[i]);
    }
    Mask(input_tensor->shape, option->end_mask, 1, parm->ends);
    ConvertShapeFormatTFLite(parm->ends);
    std::reverse(parm->ends.begin(), parm->ends.end());

    if (tf_lite_operator->inputs.size() == 4) {
        auto &strides_tensor = tf_lite_tensors[tf_lite_operator->inputs[3]];
        auto stride_size =
            tf_lite_model_buffer[strides_tensor->buffer]->data.size() / SizeofTFLiteTensorData(strides_tensor->type);
        auto stride_data_ptr = reinterpret_cast<int32_t *>(tf_lite_model_buffer[strides_tensor->buffer]->data.data());
        for (int i = 0; i < stride_size; ++i) {
            parm->strides.push_back(stride_data_ptr[i]);
        }
    } else {
        parm->strides = {(int)begin_size, 1};
    }
    ConvertShapeFormatTFLite(parm->strides);
    std::reverse(parm->strides.begin(), parm->strides.end());

    cur_layer->inputs.resize(1);
    cur_layer->inputs[0] = tf_lite_tensors[tf_lite_operator->inputs[0]]->name;
    return TNN_NS::TNN_CONVERT_OK;
}

using namespace tflite;
REGISTER_CONVERTER(StridedSlice, BuiltinOperator_STRIDED_SLICE);
}  // namespace TNN_CONVERTER
