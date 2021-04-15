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

DECLARE_OP_CONVERTER(Squeeze);

std::string OnnxSqueezeConverter::TNNOpType(const onnx::NodeProto& node, bool quantized_model) {
    return "Squeeze";
}

TNN_NS::ActivationType OnnxSqueezeConverter::ActivationType(const onnx::NodeProto& node) {
    return TNN_NS::ActivationType_None;
}

TNN_NS::Status OnnxSqueezeConverter::exec(TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource,
                                          const onnx::NodeProto& node,
                                          std::map<std::string, const onnx::TensorProto*>& proxy_initializers_map,
                                          std::map<std::string, std::shared_ptr<OnnxProxyNode>>& proxy_nodes,
                                          bool& quantized_model) {
    auto param       = new TNN_NS::SqueezeLayerParam;
    auto cur_layer   = net_structure.layers.back();
    cur_layer->param = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->type      = cur_layer->type_str;
    param->name      = cur_layer->name;
    param->quantized = false;
    param->axes      = GetAttributeIntVector(node, "axes");
    auto& data_name  = node.input(0);
    const auto& iter = proxy_initializers_map.find(data_name);
    if (iter != proxy_initializers_map.end()) {
        param->data_in_resource            = true;
        auto& resource_map                 = net_resource.resource_map;
        auto resource                      = std::make_shared<TNN_NS::SqueezeLayerResource>();
        resource_map[cur_layer->name]      = resource;
        auto data_tensor_proto             = iter->second;
        TNN_NS::RawBuffer* data_raw_buffer = nullptr;
        CreateRawBufferFromTensor(*data_tensor_proto, &data_raw_buffer);
        resource->data = *data_raw_buffer;
        cur_layer->inputs.clear();
    } else {
        param->data_in_resource = false;
    }
    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_CONVERTER(Squeeze, Squeeze);

}  // namespace TNN_CONVERTER
