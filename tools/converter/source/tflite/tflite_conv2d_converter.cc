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

DECLARE_OP_CONVERTER(Conv2D);

std::string TFLiteConv2DConverter::TNNOpType(bool quantizedModel) {
    if (quantizedModel) {
        return "QuantizedConvolution";
    }
    return "Convolution";
}

TNN_NS::Status TFLiteConv2DConverter::exec(
    TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource,
    const std::unique_ptr<tflite::OperatorT>& tf_lite_operator,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tf_lite_tensors,
    const std::vector<std::unique_ptr<tflite::BufferT>>& tf_lite_model_buffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tf_lite_op_set, bool quantizedModel) {
    TNN_NS::ConvLayerParam* param = new TNN_NS::ConvLayerParam;
    auto cur_layer                = net_structure.layers.back();

    // 3|2 inputs: input tensor, weight, (bias)
    const int input_size = tf_lite_operator->inputs.size();
    ASSERT(input_size == 2 || input_size == 3);
    const auto& conv_opt = tf_lite_operator->builtin_options.AsConv2DOptions();
    // weight index
    const int weight_index    = tf_lite_operator->inputs[1];
    const auto& weight_tensor = tf_lite_tensors[weight_index];
    // co kh kw ci
    const auto& weight_shape = weight_tensor->shape;
    ASSERT(weight_shape.size() == 4);
    const int co          = weight_shape[0];
    const int kh          = weight_shape[1];
    const int kw          = weight_shape[2];
    const int ci          = weight_shape[3];
    const int weight_size = co * kh * kw * ci;
    if (quantizedModel) {
        // TODO
    } else {
        param->name           = cur_layer->name;
        param->type           = cur_layer->type_str;
        param->quantized      = false;
        param->input_channel  = ci;
        param->output_channel = co;
        param->kernels.push_back(kw);
        param->kernels.push_back(kh);
        param->strides.push_back(conv_opt->stride_w);
        param->strides.push_back(conv_opt->stride_h);
        param->dialations.push_back(conv_opt->dilation_w_factor);
        param->dialations.push_back(conv_opt->dilation_h_factor);
        param->group = 1;
        param->pad_type = 0;
        if (conv_opt->padding == tflite::Padding_VALID) {
            // tensorflow pad valid
            param->pad_type = -1;
            param->pads.push_back(0);
            param->pads.push_back(0);
            param->pads.push_back(0);
            param->pads.push_back(0);
        } else if (conv_opt->padding == tflite::Padding_SAME) {
            param->pad_type = 0;
            param->pads.push_back(0);
            param->pads.push_back(0);
            param->pads.push_back(0);
            param->pads.push_back(0);
        }

        const auto activation = conv_opt->fused_activation_function;
        if (activation == tflite::ActivationFunctionType_RELU) {
            param->activation_type = TNN_NS::ActivationType_ReLU;
        } else if (activation == tflite::ActivationFunctionType_RELU6) {
            param->activation_type = TNN_NS::ActivationType_ReLU6;
        } else if (activation > tflite::ActivationFunctionType_NONE) {
            LOGE("MNN Conv2D do not Support fused_activation_function\n");
            return TNN_NS::TNNERR_CONVERT_UNSUPPORT_LAYER;
        }
        // update param
        cur_layer->param = std::shared_ptr<TNN_NS::LayerParam>(param);

        // weight
        auto layer_resource = new TNN_NS::ConvLayerResource;

        TNN_NS::RawBuffer filter_handle = TNN_NS::RawBuffer(weight_size * sizeof(float));
        auto original_weight_ptr =
            reinterpret_cast<const float*>(tf_lite_model_buffer[weight_tensor->buffer]->data.data());
        ConvertDataFormatTFLite(original_weight_ptr, filter_handle.force_to<float*>(), kh, kw, ci, co);
        //::memcpy(filter_handle.force_to<float*>(), original_weight_ptr, sizeof(kh*kw));
        layer_resource->filter_handle = filter_handle;
        // bias
        TNN_NS::RawBuffer bias_handle = TNN_NS::RawBuffer(co * sizeof(float));
        if (input_size == 3) {
            const auto& bias_tensor = tf_lite_tensors[tf_lite_operator->inputs[2]];
            auto bias_data_ptr = reinterpret_cast<const float*>(tf_lite_model_buffer[bias_tensor->buffer]->data.data());
            ::memcpy(bias_handle.force_to<float*>(), bias_data_ptr, sizeof(float) * co);
        }
        layer_resource->bias_handle                = bias_handle;
        net_resource.resource_map[cur_layer->name] = std::shared_ptr<TNN_NS::LayerResource>(layer_resource);
    }

    // set input output index
    cur_layer->inputs.resize(1);
    cur_layer->outputs.resize(1);
    cur_layer->inputs[0]  = tf_lite_tensors[tf_lite_operator->inputs[0]]->name;
    cur_layer->outputs[0] = tf_lite_tensors[tf_lite_operator->outputs[0]]->name;
    return TNN_NS::TNN_CONVERT_OK;
}
using namespace tflite;
REGISTER_CONVERTER(Conv2D, BuiltinOperator_CONV_2D);

}  // namespace TNN_CONVERTER