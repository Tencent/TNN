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

#include <tnn/utils/dims_vector_utils.h>

#include "tflite_op_converter.h"
#include "tflite_utils.h"

namespace TNN_CONVERTER {

DECLARE_OP_CONVERTER(Conv2D);

std::string TFLiteConv2DConverter::TNNOpType(tflite::BuiltinOperator op_code, bool quantized_model) {
    if (quantized_model) {
        return "QuantizedConvolution";
    } else {
        return "Convolution";
    }
}

tflite::ActivationFunctionType TFLiteConv2DConverter::ActivationType(
    const std::unique_ptr<tflite::OperatorT>& tf_lite_operator, tflite::BuiltinOperator op_code) {
    switch (op_code) {
        case tflite::BuiltinOperator_CONV_2D:
            return tf_lite_operator->builtin_options.AsConv2DOptions()->fused_activation_function;
        case tflite::BuiltinOperator_DEPTHWISE_CONV_2D:
            return tf_lite_operator->builtin_options.AsDepthwiseConv2DOptions()->fused_activation_function;
        default:
            return tflite::ActivationFunctionType_NONE;
    }
}
void CalculatePadSize(const std::unique_ptr<tflite::OperatorT>& tf_lite_operator,
                      const std::vector<std::unique_ptr<tflite::TensorT>>& tf_lite_tensors,
                      const tflite::BuiltinOperator tf_lite_op_type,
                      const tflite::BuiltinOptionsUnion& builtin_options_union, const int kernel_h, const int kernel_w,
                      TNN_NS::ConvLayerParam* param) {
    auto input_index          = tf_lite_operator->inputs[0];
    const auto& input_tensor  = tf_lite_tensors[input_index];
    const auto input_shape    = input_tensor->shape;
    const int input_height    = input_shape[1];
    const int input_wight     = input_shape[2];
    const auto& output_tensor = tf_lite_tensors[tf_lite_operator->outputs[0]];
    const auto& output_shape  = output_tensor->shape;
    const int output_height   = output_shape[1];
    const int output_weight   = output_shape[2];
    int pad_left              = 0;
    int pad_right             = 0;
    int pad_top               = 0;
    int pad_bottom            = 0;
    int dilation_h            = 0;
    int dilation_w            = 0;
    int stride_h              = 0;
    int stride_w              = 0;
    int kernel_extent_h       = 0;
    int kernel_extent_w       = 0;
    int total_pad_h           = 0;
    int total_pad_w           = 0;
    if (tf_lite_op_type == tflite::BuiltinOperator_CONV_2D) {
        const auto& option = builtin_options_union.AsConv2DOptions();
        dilation_h         = option->dilation_h_factor;
        dilation_w         = option->dilation_w_factor;
        stride_h           = option->stride_h;
        stride_w           = option->stride_w;
    } else if (tf_lite_op_type == tflite::BuiltinOperator_DEPTHWISE_CONV_2D) {
        const auto& option = builtin_options_union.AsDepthwiseConv2DOptions();
        dilation_h         = option->dilation_h_factor;
        dilation_w         = option->dilation_w_factor;
        stride_h           = option->stride_h;
        stride_w           = option->stride_w;
    }
    kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
    kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    total_pad_h     = (output_height - 1) * stride_h + kernel_extent_h - input_height;
    total_pad_w     = (output_weight - 1) * stride_w + kernel_extent_w - input_wight;
    pad_top         = total_pad_h % 2 == 0 ? total_pad_h / 2 : total_pad_h / 2 + 1;
    pad_bottom      = total_pad_h - pad_top;
    pad_left        = total_pad_w % 2 == 0 ? total_pad_w / 2 : total_pad_w / 2 + 1;
    pad_right       = total_pad_w - pad_left;
    param->pads.clear();
    param->pads.push_back(pad_left);
    param->pads.push_back(pad_right);
    param->pads.push_back(pad_top);
    param->pads.push_back(pad_bottom);
}

static TNN_NS::Status CreateResource(TNN_NS::NetResource& net_resource,
                                     const std::unique_ptr<tflite::OperatorT>& tf_lite_operator,
                                     const std::vector<std::unique_ptr<tflite::TensorT>>& tf_lite_tensors,
                                     const std::vector<std::unique_ptr<tflite::BufferT>>& tf_lite_model_buffer,
                                    std::string layer_name) {
    auto resource                         = std::make_shared<TNN_NS::ConvLayerResource>();
    resource->name                        = layer_name;
    net_resource.resource_map[layer_name] = resource;
    const auto& weight_tensor             = tf_lite_tensors[tf_lite_operator->inputs[1]];
    const auto& weight_quantization       = weight_tensor->quantization;
    const auto& weight_scales             = weight_quantization->scale;
    ASSERT(weight_scales.size() == 1);
    const auto& weight_zero_point = weight_quantization->zero_point;
    ASSERT(weight_zero_point.size() == 1);
    auto& weight_dims            = weight_tensor->shape;
    const int CO          = weight_dims[0];
    const int KH          = weight_dims[1];
    const int KW          = weight_dims[2];
    const int CI          = weight_dims[3];
    const int weight_size = TNN_NS::DimsVectorUtils::Count(weight_dims);
    auto filter_handle    = TNN_NS::RawBuffer(weight_size * sizeof(int8_t));
    filter_handle.SetDataType(TNN_NS::DATA_TYPE_INT8);
    auto dst     = filter_handle.force_to<int8_t*>();
    uint8_t* src = tf_lite_model_buffer[weight_tensor->buffer]->data.data();
    // OHWi -> OIHW
    for (int co = 0; co < CO; ++co) {
        for (int ci = 0; ci < CI; ++ci) {
            for (int h = 0; h < KH; ++h) {
                for (int w = 0; w < KW; ++w) {
                    dst[(co * CI + ci) * KH * KW + h * KW + w] =
                        src[(co * KH + h) * KW * CI + w * CI + ci] - weight_zero_point[0];
                }
            }
        }
    }
    resource->filter_handle = filter_handle;

    // create weight scale
    const auto& input_scale = tf_lite_tensors[tf_lite_operator->inputs[0]]->quantization->scale;
    assert(weight_scales.size() == input_scale.size());
    TNN_NS::RawBuffer scale_handle = TNN_NS::RawBuffer(weight_scales.size() * sizeof(float ));
    scale_handle.SetDataType(TNN_NS::DATA_TYPE_FLOAT);
    const auto cal_weight_scale_data = scale_handle.force_to<float*>();
    for (int i = 0; i < weight_scales.size(); ++i) {
        cal_weight_scale_data[i] = input_scale[i] * weight_scales[i];
    }
    resource->scale_handle = scale_handle;
    auto zero_point_handle = TNN_NS::RawBuffer(weight_scales.size() * sizeof(int8_t));
    zero_point_handle.SetDataType(TNN_NS::DATA_TYPE_INT8);
    resource->zero_point_handle = zero_point_handle;

    if (tf_lite_operator->inputs.size() >2) {
        // bias
        const auto& bias_tensor = tf_lite_tensors[tf_lite_operator->inputs[2]];
        const auto bias_data = reinterpret_cast<int32_t*>(tf_lite_model_buffer[bias_tensor->buffer]->data.data());
        const auto& bias_dims = bias_tensor->shape;
        const int bias_size = TNN_NS::DimsVectorUtils::Count(bias_dims);
        TNN_NS::RawBuffer bias_handle = TNN_NS::RawBuffer(bias_size * sizeof(int32_t));
        bias_handle.SetDataType(TNN_NS::DATA_TYPE_INT32);
        ::memcpy(bias_handle.force_to<int32_t*>(),bias_data, bias_size*sizeof(int32_t));
        resource->bias_handle = bias_handle;
    }

    return TNN_NS::TNN_CONVERT_OK;
}

TNN_NS::Status TFLiteConv2DConverter::exec(TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource,
                                           const std::unique_ptr<tflite::OperatorT>& tf_lite_operator,
                                           const std::vector<std::unique_ptr<tflite::TensorT>>& tf_lite_tensors,
                                           const std::vector<std::unique_ptr<tflite::BufferT>>& tf_lite_model_buffer,
                                           const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tf_lite_op_set,
                                           bool quantized_model) {
    TNN_NS::ConvLayerParam* param = new TNN_NS::ConvLayerParam;
    auto cur_layer                = net_structure.layers.back();
    cur_layer->param              = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->name                   = cur_layer->name;
    param->type                   = cur_layer->type_str;
    param->quantized              = quantized_model;
    // 3|2 inputs: input tensor, weight, (bias)
    const int input_size = tf_lite_operator->inputs.size();
    ASSERT(input_size == 2 || input_size == 3);
    param->bias = input_size > 2 ? 1 : 0;
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
    auto tf_lite_op_type  = tf_lite_op_set[tf_lite_operator->opcode_index]->builtin_code;
    if (tf_lite_op_type == tflite::BuiltinOperator_CONV_2D) {
        const auto option     = tf_lite_operator->builtin_options.AsConv2DOptions();
        param->input_channel  = ci;
        param->output_channel = co;
        param->kernels.push_back(kw);
        param->kernels.push_back(kh);
        param->strides.push_back(option->stride_w);
        param->strides.push_back(option->stride_h);
        param->dialations.push_back(option->dilation_w_factor);
        param->dialations.push_back(option->dilation_h_factor);
        param->group = 1;
        if (option->padding == tflite::Padding_VALID) {
            // tensorflow pad valid
            param->pad_type = 1;
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
        if (param->dialations[0] != 1 && param->dialations[1] != 1) {
            param->pad_type = -1;
            TNN_CONVERTER::CalculatePadSize(tf_lite_operator, tf_lite_tensors, tf_lite_op_type,
                                            tf_lite_operator->builtin_options, kh, kw, param);
        }
        const auto activation = option->fused_activation_function;
        if (activation == tflite::ActivationFunctionType_RELU) {
            param->activation_type = TNN_NS::ActivationType_ReLU;
        } else if (activation == tflite::ActivationFunctionType_RELU6) {
            param->activation_type = TNN_NS::ActivationType_ReLU6;
        } else if (activation > tflite::ActivationFunctionType_NONE) {
            LOGE("TNN Conv2D do not Support fused_activation_function\n");
            return TNN_NS::TNNERR_CONVERT_UNSUPPORT_LAYER;
        }
    } else if (tf_lite_op_type == tflite::BuiltinOperator_DEPTHWISE_CONV_2D) {
        const auto depthwise_option = tf_lite_operator->builtin_options.AsDepthwiseConv2DOptions();
        assert(co == 1);
        const int cm          = co;
        param->input_channel  = ci;
        param->output_channel = ci * cm;
        param->kernels.push_back(kw);
        param->kernels.push_back(kh);
        param->strides.push_back(depthwise_option->stride_w);
        param->strides.push_back(depthwise_option->stride_h);
        param->dialations.push_back(depthwise_option->dilation_w_factor);
        param->dialations.push_back(depthwise_option->dilation_h_factor);
        param->group    = ci;
        param->pad_type = 0;
        if (depthwise_option->padding == tflite::Padding_VALID) {
            // tensorflow pad valid
            param->pad_type = 1;
            param->pads.push_back(0);
            param->pads.push_back(0);
            param->pads.push_back(0);
            param->pads.push_back(0);
        } else if (depthwise_option->padding == tflite::Padding_SAME) {
            param->pad_type = 0;
            param->pads.push_back(0);
            param->pads.push_back(0);
            param->pads.push_back(0);
            param->pads.push_back(0);
        }
        if ((param->dialations[0] != 1 || param->dialations[1] != 1) &&
            (param->strides[0] != 1 || param->strides[1] != 1)) {
            LOGE("TFLite Converter: If any value of dilation_rate is > 1, then all values of strides must be 1.\n");
            return TNN_NS::TNNERR_INVALID_MODEL;
        }

        if ((param->dialations[0] != 1 || param->dialations[1] != 1) &&
            (param->strides[0] == 1 && param->strides[1] == 1)) {
            param->pad_type = -1;
            TNN_CONVERTER::CalculatePadSize(tf_lite_operator, tf_lite_tensors, tf_lite_op_type,
                                            tf_lite_operator->builtin_options, kh, kw, param);
        }
        const auto activation = depthwise_option->fused_activation_function;
        if (activation == tflite::ActivationFunctionType_RELU) {
            param->activation_type = TNN_NS::ActivationType_ReLU;
        } else if (activation == tflite::ActivationFunctionType_RELU6) {
            param->activation_type = TNN_NS::ActivationType_ReLU6;
        } else if (activation > tflite::ActivationFunctionType_NONE) {
            LOGE("TNN Depthwise Conv2D do not Support fused_activation_function\n");
            return TNN_NS::TNNERR_CONVERT_UNSUPPORT_LAYER;
        }
    } else {
        LOGE("TNN Conv2D do not Support operator\n");
        return TNN_NS::TNNERR_CONVERT_UNSUPPORT_LAYER;
    }

    if (quantized_model) {
        // create IntScaleResource for input
        int input_tensor_index = tf_lite_operator->inputs[0];
        TNN_NS::Status status  = CreateBlobScaleResource(net_resource, tf_lite_tensors, input_tensor_index);
        ASSERT(status == TNN_NS::TNN_CONVERT_OK);
        // create conv resource
        status = CreateResource(net_resource, tf_lite_operator, tf_lite_tensors, tf_lite_model_buffer,cur_layer->name);
        ASSERT(status == TNN_NS::TNN_CONVERT_OK);
        // create IntScaleResource for output
        int output_tensor_index = tf_lite_operator->outputs[0];
        status                  = CreateBlobScaleResource(net_resource, tf_lite_tensors, output_tensor_index);
        ASSERT(status == TNN_NS::TNN_CONVERT_OK);

    } else {
        // weight
        auto layer_resource  = new TNN_NS::ConvLayerResource;
        layer_resource->name = cur_layer->name;

        TNN_NS::RawBuffer filter_handle = TNN_NS::RawBuffer(weight_size * sizeof(float));
        auto original_weight_ptr =
            reinterpret_cast<const float*>(tf_lite_model_buffer[weight_tensor->buffer]->data.data());
        TFLiteConvertOHWI2OIHW(original_weight_ptr, filter_handle.force_to<float*>(), co, kh, kw, ci);
        layer_resource->filter_handle = ConvertRawBuffer::GetInstance()->Convert(filter_handle);
        // bias
        if (input_size == 3) {
            const auto& bias_tensor = tf_lite_tensors[tf_lite_operator->inputs[2]];
            auto bias_data_ptr = reinterpret_cast<const float*>(tf_lite_model_buffer[bias_tensor->buffer]->data.data());
            if (bias_data_ptr != nullptr) {
                TNN_NS::RawBuffer bias_handle = TNN_NS::RawBuffer(param->output_channel * sizeof(float));
                ::memcpy(bias_handle.force_to<float*>(), bias_data_ptr, param->output_channel * sizeof(float));
                layer_resource->bias_handle = ConvertRawBuffer::GetInstance()->Convert(bias_handle);
            }
        }
        net_resource.resource_map[cur_layer->name] = std::shared_ptr<TNN_NS::LayerResource>(layer_resource);
    }
    cur_layer->inputs.resize(1);
    cur_layer->inputs[0] = tf_lite_tensors[tf_lite_operator->inputs[0]]->name;
    cur_layer->outputs.resize(1);
    cur_layer->outputs[0] = tf_lite_tensors[tf_lite_operator->outputs[0]]->name;
    return TNN_NS::TNN_CONVERT_OK;
}
using namespace tflite;
REGISTER_CONVERTER(Conv2D, BuiltinOperator_CONV_2D);
REGISTER_CONVERTER(Conv2D, BuiltinOperator_DEPTHWISE_CONV_2D);

}  // namespace TNN_CONVERTER