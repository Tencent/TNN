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

#include "onnx/onnx_base_converter.h"
#include "onnx/onnx_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_CONVERTER {

DECLARE_OP_CONVERTER(Int8ConvRelu);

std::string OnnxInt8ConvReluConverter::TNNOpType(const onnx::NodeProto &node, bool quantized_model) {
    return "QuantizedConvolution";
}

TNN_NS::ActivationType OnnxInt8ConvReluConverter::ActivationType(const onnx::NodeProto &node) {
    return TNN_NS::ActivationType_ReLU;
}

TNN_NS::Status OnnxInt8ConvReluConverter::exec(tnn::NetStructure &net_structure, tnn::NetResource &net_resource,
                                               const onnx::NodeProto &node,
                                               std::map<std::string, const onnx::TensorProto *> proxy_initializers_map,
                                               std::map<std::string, std::shared_ptr<OnnxProxyNode>> proxy_nodes,
                                               bool &quantized_model) {
    TNN_NS::ConvLayerParam *param = new TNN_NS::ConvLayerParam;
    auto cur_layer                = net_structure.layers.back();
    cur_layer->param              = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->name                   = cur_layer->name;
    param->type                   = cur_layer->type_str;
    param->quantized              = true;
    const int input_size          = node.input_size();
    ASSERT(input_size == 2 || input_size == 3);
    // Get weight
    const auto &weight_name = node.input(1);
    const auto &weight_node = FindNodeProto(weight_name, proxy_nodes);
    auto weight_shape       = GetAttributeIntVector(*weight_node, "shape");
    assert(weight_shape.size() == 4);
    const int64_t co             = weight_shape[0];
    const int64_t kh             = weight_shape[1];
    const int64_t kw             = weight_shape[2];
    const int64_t ci             = weight_shape[3];
    const int64_t weight_count   = co * kw * kw * ci;
    auto raw_wight_value_strings = GetAttributeString(*weight_node, "values", "");
    //    auto weight_value_strings = SplitString(raw_wight_value_strings, ",");
    //    std::vector<int64_t> weight_value;
    //    for (const auto& iter : weight_value_strings) {
    //        weight_value.push_back(std::stoll(iter));
    //    }
    //    assert(weight_value.size() == weight_count );
    auto weight_scale      = GetAttributeFloat(*weight_node, "Y_scale", 1.0);
    auto weight_zero_point = GetAttributeInt(*weight_node, "Y_zero_point", 0);

    param->input_channel  = ci;
    param->output_channel = co;
    param->kernels.push_back(kw);
    param->kernels.push_back(kh);
    // onnx order: stride_h, stride_w
    // tnn  order: stride_w, stride_h
    auto strides = GetAttributeIntVector(node, "strides");
    ASSERT(strides.size() == 2);
    param->strides = {(int)strides[1], (int)strides[0]};
    // dilation
    auto dilations = GetAttributeIntVector(node, "dilations");
    ASSERT(dilations.size() == 2);
    param->dialations = {(int)dilations[1], (int)dilations[0]};
    param->group      = 1;
    param->pad_type   = 0;
    auto pads         = GetAttributeIntVector(node, "pads");
    param->pads       = {(int)pads[0], (int)pads[1], (int)pads[2], (int)pads[3]};
    ASSERT(pads.size() == 4);
    param->activation_type = TNN_NS::ActivationType_ReLU;
    // weight
    auto layer_resource             = new TNN_NS::ConvLayerResource;
    layer_resource->name            = cur_layer->name;
    TNN_NS::RawBuffer filter_handle = TNN_NS::RawBuffer(weight_count * sizeof(int32_t));

    if (input_size > 2) {
        // Get Bias
        const auto &bias_name = node.input(2);
        const auto &bias_node = FindNodeProto(bias_name, proxy_nodes);
        auto bias_shape       = GetAttributeIntVector(*bias_node, "shape");
        assert(bias_shape.size() == 1);
        auto bias_value      = GetAttributeIntVector(*bias_node, "values");
        auto bias_scale      = GetAttributeFloat(*bias_node, "Y_scale", 1.0);
        auto bias_zero_point = GetAttributeInt(*bias_node, "Y_zero_point", 0);
    }
    cur_layer->inputs.resize(1);
    cur_layer->inputs[0] = node.input(0);
    cur_layer->outputs.resize(1);
    cur_layer->outputs[0] = node.output(0);
    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_CONVERTER(Int8ConvRelu, Int8ConvRelu);

}  // namespace TNN_CONVERTER