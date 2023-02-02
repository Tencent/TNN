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

#include "onnx/onnx_utils.h"
#include "tools/converter/source/onnx/onnx_base_converter.h"

namespace TNN_CONVERTER {
DECLARE_OP_CONVERTER(Conv);

std::string OnnxConvConverter::TNNOpType(const onnx::NodeProto &node, bool quantized_model) {
    return "Convolution";
}
TNN_NS::ActivationType OnnxConvConverter::ActivationType(const onnx::NodeProto &node) {
    return TNN_NS::ActivationType_None;
}
TNN_NS::Status OnnxConvConverter::exec(TNN_NS::NetStructure &net_structure, TNN_NS::NetResource &net_resource,
                                       const onnx::NodeProto &node,
                                       std::map<std::string, const onnx::TensorProto *> &proxy_initializers_map,
                                       std::map<std::string, std::shared_ptr<OnnxProxyNode>> &proxy_nodes,
                                       bool &quantized_model) {
    TNN_NS::ConvLayerParam *param = new TNN_NS::ConvLayerParam;
    auto cur_layer                = net_structure.layers.back();
    cur_layer->param              = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->name                   = cur_layer->name;
    param->type                   = cur_layer->type_str;
    param->quantized              = false;

    // 3|2 inputs: input tensor, weight, (bias)
    const int input_size = node.input_size();
    ASSERT(input_size == 2 || input_size == 3);

    const int has_bias = input_size == 3 ? 1 : 0;

    const auto &filter_name   = node.input(1);
    const auto *filter_tensor = proxy_initializers_map[filter_name];
    const int co              = filter_tensor->dims(0);
    const int ci              = filter_tensor->dims(1);
    const int kh              = filter_tensor->dims(2);
    const int kw              = filter_tensor->dims(3);
    param->bias               = has_bias;
    param->input_channel      = ci;
    param->output_channel     = co;
    param->kernels            = {kw, kh};
    auto strides              = GetAttributeIntVector(node, "strides");
    ASSERT(strides.size() == 2);
    param->strides = {strides[1], strides[0]};
    // dilation
    auto dilations = GetAttributeIntVector(node, "dilations");
    ASSERT(dilations.size() == 2);
    param->dialations = {dilations[1], dilations[0]};
    param->group      = GetAttributeInt(node, "group", 1);
    // parse pads type
    auto pads = GetAttributeIntVector(node, "pads");
    if (!pads.empty()) {
        param->pad_type = -1;
        if (pads[0] < pads[2] || pads[1] < pads[3]) {
            // same upper
            param->pad_type = 0;
        }
        param->pads = {pads[1], pads[3], pads[0], pads[2]};
    } else {
        auto auto_pad = GetAttributeString(node, "auto_pad", "NOTSET");
        if (auto_pad == "NOTSET") {
            param->pad_type = -1;
        } else if (auto_pad == "SAME_UPPER") {
            param->pad_type = 0;
        } else if (auto_pad == "VALID") {
            param->pad_type = 1;
        } else {
            LOGE("Conv: SAME_LOWER does not support, change toSAME_UPPER\n");
            return TNN_NS::TNNERR_CONVERT_UNSUPPORT_LAYER;
        }
        param->pads = {0, 0, 0, 0};
    }
    ASSERT(pads.size() == 4);

    param->activation_type = TNN_NS::ActivationType_None;

    // weight
    auto layer_resource             = new TNN_NS::ConvLayerResource;
    layer_resource->name            = cur_layer->name;
    const int weight_count          = co * kh * kw * ci;
    TNN_NS::RawBuffer filter_handle = TNN_NS::RawBuffer(weight_count * sizeof(float));
    filter_handle.SetDataType(TNN_NS::DATA_TYPE_FLOAT);
    filter_handle.SetBufferDims({co, ci, kh, kw});
    auto *filter_tensor_data = reinterpret_cast<const float *>(GetTensorProtoData(*filter_tensor));
    ::memcpy(filter_handle.force_to<float *>(), filter_tensor_data, weight_count * sizeof(float));
    layer_resource->filter_handle = filter_handle;
    // bias
    if (has_bias) {
        TNN_NS::RawBuffer bias_handle = TNN_NS::RawBuffer(co * sizeof(float));
        bias_handle.SetDataType(TNN_NS::DATA_TYPE_FLOAT);
        bias_handle.SetBufferDims({co});
        const auto &bias_name   = node.input(2);
        const auto *bias_tensor = proxy_initializers_map[bias_name];
        auto *bias_tensor_data  = reinterpret_cast<const float *>(GetTensorProtoData(*bias_tensor));
        ::memcpy(bias_handle.force_to<float *>(), bias_tensor_data, co * sizeof(float));
        layer_resource->bias_handle = bias_handle;
    }
    net_resource.resource_map[cur_layer->name] = std::shared_ptr<TNN_NS::LayerResource>(layer_resource);

    cur_layer->inputs.resize(1);
    cur_layer->inputs[0] = node.input(0);
    cur_layer->outputs.resize(1);
    cur_layer->outputs[0] = node.output(0);

    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_CONVERTER(Conv, Conv);
}  // namespace TNN_CONVERTER
