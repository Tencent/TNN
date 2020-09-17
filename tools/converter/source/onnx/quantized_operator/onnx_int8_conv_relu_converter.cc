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
    auto weight_shape       = GetAttributeIntVector(node, "shape");
    assert(weight_shape.size() == 4);
    const int64_t co       = weight_shape[0];
    const int64_t kh       = weight_shape[1];
    const int64_t kw       = weight_shape[2];
    const int64_t ci       = weight_shape[3];
    auto weight_value      = GetAttributeIntVector(node, "value");
    auto weight_scale      = GetAttributeFloat(node, "Y_scale", 1.0);
    auto weight_zero_point = GetAttributeInt(node, "Y_zero_point", 0);
    if (input_size > 2) {
        // Get Bias
        const auto &bias_name = node.input(2);
        const auto &bias_node = FindNodeProto(weight_name, proxy_nodes);
        auto bias_shape       = GetAttributeIntVector(node, "shape");
        assert(bias_shape.size() == 1);
        auto bias_value      = GetAttributeIntVector(node, "value");
        auto bias_scale      = GetAttributeFloat(node, "Y_scale", 1.0);
        auto bias_zero_point = GetAttributeInt(node, "Y_zero_point", 0);
    }

    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_CONVERTER(Int8ConvRelu, Int8ConvRelu);

}  // namespace TNN_CONVERTER