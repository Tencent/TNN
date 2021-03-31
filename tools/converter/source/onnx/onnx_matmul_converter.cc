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
DECLARE_OP_CONVERTER(MatMul);

std::string OnnxMatMulConverter::TNNOpType(const onnx::NodeProto &node, bool quantized_model) {
    return "MatMul";
}

TNN_NS::ActivationType OnnxMatMulConverter::ActivationType(const onnx::NodeProto &node) {
    return TNN_NS::ActivationType_None;
}

TNN_NS::Status OnnxMatMulConverter::exec(TNN_NS::NetStructure &net_structure, TNN_NS::NetResource &net_resource,
                                         const onnx::NodeProto &node,
                                         std::map<std::string, const onnx::TensorProto *> &proxy_initializers_map,
                                         std::map<std::string, std::shared_ptr<OnnxProxyNode>> &proxy_nodes,
                                         bool &quantized_model) {
    auto param       = new TNN_NS::MatMulLayerParam;
    auto cur_layer   = net_structure.layers.back();
    cur_layer->param = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->type      = cur_layer->type_str;
    param->name      = cur_layer->name;
    param->quantized = false;

    std::pair<int, std::string> weight_input_index_name;
    int weight_input_index = -1;
    std::string weight_name;
    auto status = GetWeightInputIndexName(weight_input_index, weight_name, node, proxy_initializers_map, proxy_nodes);
    if (status != TNN_NS::TNN_CONVERT_OK) {
        return status;
    }

    param->weight_position = weight_input_index;
    if (weight_input_index == -1) {
        return TNN_NS::TNN_CONVERT_OK;
    }

    const auto *weight_tensor = proxy_initializers_map[weight_name];
    auto *weight_tensor_data  = reinterpret_cast<const float *>(GetTensorProtoData(*weight_tensor));
    const auto &weight_shape  = weight_tensor->dims();
    int weight_size           = 1;
    TNN_NS::DimsVector weight_dims;
    for (const auto dim : weight_shape) {
        weight_size *= dim;
        weight_dims.push_back(dim);
    }

    auto layer_resource             = new TNN_NS::MatMulLayerResource;
    layer_resource->name            = cur_layer->name;
    TNN_NS::RawBuffer weight_handle = TNN_NS::RawBuffer(weight_size * sizeof(float));
    weight_handle.SetBufferDims(weight_dims);
    ::memcpy(weight_handle.force_to<float *>(), weight_tensor_data, weight_size * sizeof(float));
    layer_resource->weight                     = weight_handle;
    net_resource.resource_map[cur_layer->name] = std::shared_ptr<TNN_NS::LayerResource>(layer_resource);

    cur_layer->inputs.resize(1);
    if (weight_input_index == 0) {
        cur_layer->inputs[0] = node.input(1);
    } else {
        cur_layer->inputs[0] = node.input(0);
    }

    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_CONVERTER(MatMul, MatMul);

}  // namespace TNN_CONVERTER
