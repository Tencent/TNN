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

#include "tnn/utils/dims_utils.h"
#include "torch/torch_base_converter.h"
#include "torch/torch_utils.h"

namespace TNN_CONVERTER {

DECLARE_TORCH_OP_CONVERTER(Binary);

std::string TorchBinaryConverter::TNNOpType(const torch::jit::Node *node, bool quantized_model) {
    std::string type = node->kind().toQualString();
    if (type == "aten::add") {
        return "Add";
    } else {
        LOGE("TorchBinaryConverter does not support type %s", type.c_str());
        return "";
    }
}

TNN_NS::ActivationType TorchBinaryConverter::ActivationType(const torch::jit::Node *node) {
    return TNN_NS::ActivationType_None;
}
// add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
// add_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
TNN_NS::Status TorchBinaryConverter::exec(tnn::NetStructure &net_structure, tnn::NetResource &net_resource,
                                          const torch::jit::Node *node, bool quantized_mode) {
    auto cur_layer      = std::make_shared<TNN_NS::LayerInfo>();
    cur_layer->name     = node->output(0)->debugName();
    auto type_name      = TNNOpType(node, quantized_mode);
    auto layer_type     = TNN_NS::GlobalConvertLayerType(type_name);
    cur_layer->type     = layer_type;
    cur_layer->type_str = type_name;

    cur_layer->inputs.push_back(node->input(0)->debugName());
    cur_layer->outputs.push_back(node->output(0)->debugName());
    net_structure.layers.push_back(cur_layer);
    // parse param
    auto param         = new TNN_NS::MultidirBroadcastLayerParam;
    cur_layer->param   = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->type        = cur_layer->type_str;
    param->name        = cur_layer->name;
    param->quantized   = false;
    const auto &inputs = node->inputs();
    for (int i = 1; i < inputs.size() - 1; ++i) {
        const auto &value_kind = inputs[i]->type()->kind();
        if (value_kind == c10::TypeKind::TensorType) {
            cur_layer->inputs.push_back(inputs[i]->debugName());
            param->weight_input_index = -1;
        } else {
            param->weight_input_index                       = 1;
            auto *layer_resource                            = new TNN_NS::EltwiseLayerResource;
            layer_resource->name                            = cur_layer->name;
            layer_resource->element_handle                  = CreateRawBufferFromValue(node->input(1));
            layer_resource->element_shape                   = layer_resource->element_handle.GetBufferDims();
            net_resource.resource_map[layer_resource->name] = std::shared_ptr<TNN_NS::LayerResource>(layer_resource);
        }
    }
    // parse resource

    return TNN_NS::TNN_CONVERT_OK;
}

// add_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
REGISTER_TORCH_OP_CONVERTER(Binary, aten, add);
REGISTER_TORCH_OP_CONVERTER(Binary, aten, floor_divide);

}  // namespace TNN_CONVERTER
