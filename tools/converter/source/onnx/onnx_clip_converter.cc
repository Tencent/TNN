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
#include "tnn/interpreter/tnn/objseri.h"
#include "tools/converter/source/onnx/onnx_base_converter.h"

namespace TNN_CONVERTER {
DECLARE_OP_CONVERTER(Clip);

std::string OnnxClipConverter::TNNOpType(const onnx::NodeProto &node, bool quantized_model) {
    return "Clip";
}

TNN_NS::ActivationType OnnxClipConverter::ActivationType(const onnx::NodeProto &node) {
    return TNN_NS::ActivationType_None;
}

TNN_NS::Status OnnxClipConverter::exec(TNN_NS::NetStructure &net_structure, TNN_NS::NetResource &net_resource,
                                       const onnx::NodeProto &node,
                                       std::map<std::string, const onnx::TensorProto *> &proxy_initializers_map,
                                       std::map<std::string, std::shared_ptr<OnnxProxyNode>> &proxy_nodes,
                                       bool &quantized_model) {
    const auto min = GetAttributeFloat(node, "min", 1, -DBL_MAX, proxy_initializers_map);
    const auto max = GetAttributeFloat(node, "max", 2, DBL_MAX, proxy_initializers_map);
    auto cur_layer = net_structure.layers.back();
    if (std::fabs(min) <= DBL_EPSILON && std::fabs(max - 6) <= DBL_EPSILON) {
        cur_layer->type_str = "ReLU6";
        cur_layer->type     = TNN_NS::LAYER_RELU6;
        cur_layer->param    = std::shared_ptr<TNN_NS::LayerParam>(new TNN_NS::LayerParam);
        return TNN_NS::TNN_CONVERT_OK;
    }

    const std::string &onnx_op = node.op_type();
    auto param                 = new TNN_NS::ClipLayerParam;
    cur_layer->param           = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->type                = cur_layer->type_str;
    param->name                = cur_layer->name;
    param->quantized           = false;
    param->min                 = min;
    param->max                 = max;

    cur_layer->inputs.resize(1);
    cur_layer->inputs[0] = node.input(0);

    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_CONVERTER(Clip, Clip);

}  // namespace TNN_CONVERTER
