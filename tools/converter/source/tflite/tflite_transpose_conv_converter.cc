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

DECLARE_OP_CONVERTER(TransposeConv);

std::string TFLiteTransposeConvConverter::TNNOpType(tflite::BuiltinOperator op_code, bool quantized_model) {
    return "Deconvolution";
}

tflite::ActivationFunctionType TFLiteTransposeConvConverter::ActivationType(
    const std::unique_ptr<tflite::OperatorT> &tf_lite_operator, tflite::BuiltinOperator op_code) {
    return tflite::ActivationFunctionType_NONE;
}

TNN_NS::Status TFLiteTransposeConvConverter::exec(
    TNN_NS::NetStructure &net_structure, TNN_NS::NetResource &net_resource,
    const std::unique_ptr<tflite::OperatorT> &tf_lite_operator,
    const std::vector<std::unique_ptr<tflite::TensorT>> &tf_lite_tensors,
    const std::vector<std::unique_ptr<tflite::BufferT>> &tf_lite_model_buffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tf_lite_op_set, bool quantized_model) {
    TNN_NS::ConvLayerParam *param = new TNN_NS::ConvLayerParam;
    auto cur_layer                = net_structure.layers.back();
    cur_layer->param              = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->name                   = cur_layer->name;
    param->type                   = cur_layer->type_str;
    param->quantized              = false;
    // 3|2 inputs: input tensor, weight, (bias)
    const int input_size = tf_lite_operator->inputs.size();
    ASSERT(input_size == 2 || input_size == 3);
    // weight index
    const int weight_index    = tf_lite_operator->inputs[1];
    const auto &weight_tensor = tf_lite_tensors[weight_index];
    // co kh kw ci
    const auto &weight_shape = weight_tensor->shape;
    ASSERT(weight_shape.size() == 4);
    int co                = weight_shape[0];
    int kh                = weight_shape[1];
    int kw                = weight_shape[2];
    int ci                = weight_shape[3];
    int weight_size       = co * kh * kw * ci;
    const auto option     = tf_lite_operator->builtin_options.AsTransposeConvOptions();
    param->input_channel  = ci;
    param->output_channel = co;
    param->kernels.push_back(kw);
    param->kernels.push_back(kh);
    param->strides.push_back(option->stride_w);
    param->strides.push_back(option->stride_h);

    param->dialations.push_back(1);
    param->dialations.push_back(1);
    param->group    = 1;
    param->pad_type = 0;
    if (option->padding == tflite::Padding_VALID) {
        // tensorflow pad valid
        param->pad_type = -1;
        param->pads.push_back(0);
        param->pads.push_back(0);
        param->pads.push_back(0);
        param->pads.push_back(0);
    } else if (option->padding == tflite::Padding_SAME) {
        param->pad_type = 0;
        param->pads.push_back(0);
        param->pads.push_back(0);
        param->pads.push_back(0);
        param->pads.push_back(0);
    }
    param->activation_type = TNN_NS::ActivationType_None;

    auto layer_resource  = new TNN_NS::ConvLayerResource;
    layer_resource->name = cur_layer->name;

    TNN_NS::RawBuffer filter_handle = TNN_NS::RawBuffer(weight_size * sizeof(float));
    auto original_weight_ptr =
        reinterpret_cast<const float *>(tf_lite_model_buffer[weight_tensor->buffer]->data.data());
    TFLiteConvertOHWI2IOHW(original_weight_ptr, filter_handle.force_to<float *>(), co, kh, kw, ci);
    layer_resource->filter_handle              = filter_handle;
    net_resource.resource_map[cur_layer->name] = std::shared_ptr<TNN_NS::LayerResource>(layer_resource);

    cur_layer->inputs.resize(1);
    cur_layer->inputs[0] = tf_lite_tensors[tf_lite_operator->inputs[2]]->name;
    cur_layer->outputs.resize(1);
    cur_layer->outputs[0] = tf_lite_tensors[tf_lite_operator->outputs[0]]->name;
    return TNN_NS::TNN_CONVERT_OK;
}

using namespace tflite;
REGISTER_CONVERTER(TransposeConv, BuiltinOperator_TRANSPOSE_CONV);
}  // namespace TNN_CONVERTER
