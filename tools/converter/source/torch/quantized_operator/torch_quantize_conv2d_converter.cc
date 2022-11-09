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

#include "ATen/native/quantized/cpu/conv_packed_params.h"
#include "tnn/interpreter/tnn/objseri.h"
#include "tnn/utils/dims_utils.h"
#include "torch/torch.h"
#include "torch/torch_base_converter.h"
#include "torch/torch_utils.h"

namespace TNN_CONVERTER {

DECLARE_TORCH_OP_CONVERTER(QuantizedConv2d);

std::string TorchQuantizedConv2dConverter::TNNOpType(const torch::jit::Node *node, bool quantized_model) {
    return "QuantizedConvolution";
}

TNN_NS::ActivationType TorchQuantizedConv2dConverter::ActivationType(const torch::jit::Node *node) {
    return TNN_NS::ActivationType_None;
}

static TNN_NS::RawBuffer CreateFilterScale(const at::Tensor &filter) {
    TNN_NS::RawBuffer filter_scale_handle;
    if (filter.qscheme() == c10::kPerChannelAffine || filter.qscheme() == c10::kPerChannelSymmetric) {
        at::Tensor q_scale_tensor = filter.q_per_channel_scales();
        filter_scale_handle       = CreateRawBufferFromTensor(q_scale_tensor);
    } else if (filter.qscheme() == c10::kPerTensorAffine || filter.qscheme() == c10::kPerTensorSymmetric) {
        float q_scale       = filter.q_scale();
        filter_scale_handle = TNN_NS::RawBuffer(sizeof(float), reinterpret_cast<char *>(&q_scale), {1});
        filter_scale_handle.SetDataType(TNN_NS::DATA_TYPE_FLOAT);
    } else {
        LOGE("TorchQuantizedConv2dConverter does not support weight scale type\n");
        ASSERT(0);
    }
    return filter_scale_handle;
}

static TNN_NS::RawBuffer CreateFilterZp(const at::Tensor &filter) {
    TNN_NS::RawBuffer filter_zp_handle;
    if (filter.qscheme() == c10::kPerChannelAffine || filter.qscheme() == c10::kPerChannelSymmetric) {
        at::Tensor q_zp_tensor = filter.q_per_channel_scales();
        filter_zp_handle       = CreateRawBufferFromTensor(q_zp_tensor);
        filter_zp_handle.SetBufferDims({filter_zp_handle.GetDataCount()});
    } else if (filter.qscheme() == c10::kPerTensorAffine || filter.qscheme() == c10::kPerTensorSymmetric) {
        int8_t q_zp      = (int8_t)filter.q_zero_point();
        filter_zp_handle = TNN_NS::RawBuffer(sizeof(int8_t), reinterpret_cast<char *>(&q_zp), {1});
        filter_zp_handle.SetDataType(TNN_NS::DATA_TYPE_INT8);
    } else {
        LOGE("TorchQuantizedConv2dConverter does not support weight zp type\n");
        ASSERT(0);
    }
    return filter_zp_handle;
}

static TNN_NS::RawBuffer FuseInputScaleToFilterScale(TNN_NS::RawBuffer &input_scale, TNN_NS::RawBuffer &filter_scale) {
    TNN_NS::RawBuffer fused_file_scale(filter_scale);

    float *filter_scale_ptr       = filter_scale.force_to<float *>();
    const int filter_count        = filter_scale.GetDataCount();
    float *input_scale_ptr        = input_scale.force_to<float *>();
    const int input_scale_count   = input_scale.GetDataCount();
    float *fused_filter_scale_ptr = fused_file_scale.force_to<float *>();

    for (int i = 0; i < filter_count; ++i) {
        int input_scale_idx       = input_scale_count == 1 ? 0 : i;
        fused_filter_scale_ptr[i] = input_scale_ptr[input_scale_idx] * filter_scale_ptr[i];
    }
    return fused_file_scale;
}

static TNN_NS::RawBuffer QuantizedBias(TNN_NS::RawBuffer &bias_handle, TNN_NS::RawBuffer &fused_filter_scale) {
    const int bias_count   = bias_handle.GetDataCount();
    const int filter_count = fused_filter_scale.GetDataCount();
    float *bias_data_ptr   = bias_handle.force_to<float *>();
    float *filter_data_ptr = fused_filter_scale.force_to<float *>();
    TNN_NS::RawBuffer quantized_bias_handle(bias_count * sizeof(int32_t));
    quantized_bias_handle.SetDataType(TNN_NS::DATA_TYPE_INT32);
    quantized_bias_handle.SetBufferDims({bias_count});
    int32_t *quantized_bias_data_ptr = quantized_bias_handle.force_to<int32_t *>();

    for (int i = 0; i < bias_count; ++i) {
        int filter_scale_index     = filter_count == 1 ? 0 : i;
        quantized_bias_data_ptr[i] = std::nearbyint(bias_data_ptr[i] / filter_data_ptr[filter_scale_index]);
    }
    return quantized_bias_handle;
}

/**
 * conv:
 *      input_name
 *      Conv2dPackedParamsBase
 *      output scale
 *      output zp
 * */
TNN_NS::Status TorchQuantizedConv2dConverter::exec(tnn::NetStructure &net_structure, tnn::NetResource &net_resource,
                                                   const torch::jit::Node *node, bool quantized_mode) {
    auto cur_layer          = std::make_shared<TNN_NS::LayerInfo>();
    cur_layer->name         = node->output(0)->debugName();
    auto type_name          = TNNOpType(node, quantized_mode);
    auto layer_type         = TNN_NS::GlobalConvertLayerType(type_name);
    cur_layer->type         = layer_type;
    cur_layer->type_str     = type_name;
    std::string input_name  = node->input(0)->debugName();
    std::string output_name = node->output(0)->debugName();
    cur_layer->inputs.push_back(input_name);
    cur_layer->outputs.push_back(node->output(0)->debugName());
    net_structure.layers.push_back(cur_layer);

    auto param       = new TNN_NS::ConvLayerParam;
    cur_layer->param = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->type      = cur_layer->type_str;
    param->name      = cur_layer->name;
    param->quantized = true;

    const auto &inputs             = node->inputs();
    const auto conv_value          = torch::jit::toIValue(inputs[1]).value();
    const auto slots               = conv_value.toObject().get()->slots();
    const auto conv_packed_param   = reinterpret_cast<ConvPackedParamsBase<2> *>(slots[0].toCapsule().get());
    const auto filter_and_bias     = conv_packed_param->unpack();
    at::Tensor filter              = std::get<0>(filter_and_bias);
    c10::optional<at::Tensor> bias = std::get<1>(filter_and_bias);
    param->group                   = (int)conv_packed_param->groups();
    const auto &filter_dims        = GetDimsFromTensor(filter);
    const int co                   = filter_dims[0];
    const int ci                   = filter_dims[1];
    const int kh                   = filter_dims[2];
    const int kw                   = filter_dims[3];
    param->input_channel           = ci;
    param->output_channel          = co;
    param->kernels.push_back(kw);
    param->kernels.push_back(kh);
    // torch order: stride_h, stride_w
    // tnn   order: stride_w, stride_h
    const auto &stride   = conv_packed_param->stride();
    const auto &padding  = conv_packed_param->padding();
    const auto &dilation = conv_packed_param->dilation();
    ASSERT(stride.size() == 2 && dilation.size() == 2 && padding.size() == 2);
    param->strides.push_back(stride[1]);
    param->strides.push_back(stride[0]);
    param->dialations.push_back(dilation[1]);
    param->dialations.push_back(dilation[0]);
    // torch order: [pad_h, pad_w]
    // tnn   order: [w_begin w_end h_begin h_end d_begin d_end]
    param->pad_type = -1;
    param->pads.push_back(padding[1]);
    param->pads.push_back(padding[1]);
    param->pads.push_back(padding[0]);
    param->pads.push_back(padding[0]);
    std::string torch_op_type = node->kind().toQualString();
    if (torch_op_type == "quantized::conv2d_relu") {
        param->activation_type = TNN_NS::ActivationType_ReLU;
    } else {
        param->activation_type = TNN_NS::ActivationType_None;
    }
    std::string input_blob_scale_name = input_name + BLOB_SCALE_SUFFIX;


    // parse resource
    auto &resource_map   = net_resource.resource_map;
    auto *layer_resource = new TNN_NS::ConvLayerResource;
    layer_resource->name = cur_layer->name;
    // input blob scale
    if (resource_map.find(input_blob_scale_name) == resource_map.end()) {
        LOGE("TorchQuantizedConv2dConverter can not get input blob scale\n");
        return TNN_NS::TNNERR_CONVERT_UNSUPPORT_LAYER;
    }
    const auto &input_blob_scale = dynamic_cast<TNN_NS::IntScaleResource *>(resource_map[input_blob_scale_name].get());
    // TODO: order ?
    // do not parse filter zp
    layer_resource->filter_handle = CreateRawBufferFromTensor(filter);
    layer_resource->scale_handle  = CreateFilterScale(filter);
    layer_resource->scale_handle =
        FuseInputScaleToFilterScale(input_blob_scale->scale_handle, layer_resource->scale_handle);

    if (bias.has_value()) {
        // quantized float -> int32_t
        layer_resource->bias_handle = CreateRawBufferFromTensor(bias.value());
        if (layer_resource->bias_handle.GetDataType() == tnn::DATA_TYPE_FLOAT) {
            layer_resource->bias_handle = QuantizedBias(layer_resource->bias_handle, layer_resource->scale_handle);
        }
    }

    // output blob scale
    std::string output_blob_cale_name = output_name + BLOB_SCALE_SUFFIX;
    if (resource_map.find(output_blob_cale_name) == resource_map.end()) {
        auto output_blob_scale_resource               = new TNN_NS::IntScaleResource;
        output_blob_scale_resource->name              = output_blob_cale_name;
        output_blob_scale_resource->scale_handle      = CreateRawBufferFromValue(inputs[2]);
        output_blob_scale_resource->zero_point_handle = CreateRawBufferFromValue(inputs[3]);
        net_resource.resource_map[output_blob_cale_name] =
            std::shared_ptr<TNN_NS::LayerResource>(output_blob_scale_resource);
    }
    net_resource.resource_map[layer_resource->name] = std::shared_ptr<TNN_NS::LayerResource>(layer_resource);

    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_TORCH_OP_CONVERTER(QuantizedConv2d, quantized, conv2d_relu);
REGISTER_TORCH_OP_CONVERTER(QuantizedConv2d, quantized, conv2d);

}  // namespace TNN_CONVERTER