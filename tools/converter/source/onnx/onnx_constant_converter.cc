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

#include "onnx_base_converter.h"
#include "onnx_utils.h"

namespace TNN_CONVERTER {

DECLARE_OP_CONVERTER(Constant);

std::string OnnxConstantConverter::TNNOpType(const onnx::NodeProto &node, bool quantized_model) {
    return "Const";
}

TNN_NS::ActivationType OnnxConstantConverter::ActivationType(const onnx::NodeProto &node) {
    return TNN_NS::ActivationType_None;
}

TNN_NS::Status OnnxConstantConverter::exec(TNN_NS::NetStructure &net_structure, TNN_NS::NetResource &net_resource,
                                           const onnx::NodeProto &node,
                                           std::map<std::string, const onnx::TensorProto *> &proxy_initializers_map,
                                           std::map<std::string, std::shared_ptr<OnnxProxyNode>> &proxy_nodes,
                                           bool &quantized_model) {
    auto param                      = new TNN_NS::ConstLayerParam;
    auto cur_layer                  = net_structure.layers.back();
    cur_layer->param                = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->name                     = cur_layer->name;
    param->type                     = cur_layer->type_str;
    param->quantized                = false;
    const onnx::TensorProto *tensor = GetTensorFromConstantNode(node);
    ASSERT(tensor != nullptr);
    TNN_NS::DimsVector dims             = CreateDimsVectorFromTensor(*tensor);
    param->dims                         = dims;
    TNN_NS::RawBuffer *const_raw_buffer = nullptr;
    CreateRawBufferFromTensor(*tensor, &const_raw_buffer);
    ASSERT(const_raw_buffer != nullptr);
    auto const_layer_resource                 = new TNN_NS::ConstLayerResource;
    const_layer_resource->weight_handle       = *const_raw_buffer;
    const_layer_resource->name                = cur_layer->name;
    net_resource.constant_map[node.output(0)] = std::shared_ptr<TNN_NS::RawBuffer>(const_raw_buffer);
    // net_resource.resource_map[cur_layer->name] = std::shared_ptr<TNN_NS::LayerResource>(const_layer_resource);

    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_CONVERTER(Constant, Constant);

}  // namespace TNN_CONVERTER
