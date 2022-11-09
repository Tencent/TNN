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

DECLARE_TORCH_OP_CONVERTER(Addmm);

std::string TorchAddmmConverter::TNNOpType(const torch::jit::Node *node, bool quantized_model) {
    return "InnerProduct";
}

TNN_NS::ActivationType TorchAddmmConverter::ActivationType(const torch::jit::Node *node) {
    return TNN_NS::ActivationType_None;
}
// aten::addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) ->
// Tensor(a!) aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
TNN_NS::Status TorchAddmmConverter::exec(tnn::NetStructure &net_structure, tnn::NetResource &net_resource,
                                         const torch::jit::Node *node, bool quantized_mode) {
    auto cur_layer      = std::make_shared<TNN_NS::LayerInfo>();
    cur_layer->name     = node->output(0)->debugName();
    auto type_name      = TNNOpType(node, quantized_mode);
    auto layer_type     = TNN_NS::GlobalConvertLayerType(type_name);
    cur_layer->type     = layer_type;
    cur_layer->type_str = type_name;
    net_structure.layers.push_back(cur_layer);
    // parse param
    auto param         = new TNN_NS::InnerProductLayerParam;
    cur_layer->param   = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->type        = cur_layer->type_str;
    param->name        = cur_layer->name;
    param->quantized   = false;
    param->has_bias    = 1;
    const auto &inputs = node->inputs();

    cur_layer->inputs.push_back(inputs[1]->debugName());
    cur_layer->outputs.push_back(node->output(0)->debugName());

    auto *layer_resource            = new TNN_NS::InnerProductLayerResource;
    layer_resource->name            = cur_layer->name;
    layer_resource->bias_handle     = CreateRawBufferFromValue(inputs[0]);
    TNN_NS::RawBuffer weight_handle = CreateRawBufferFromValue(inputs[2]);
    TNN_NS::DimsVector weight_dims  = weight_handle.GetBufferDims();
    ASSERT(weight_dims.size() == 2);
    weight_handle.Permute(weight_dims[0], weight_dims[1]);
    weight_handle.SetBufferDims({weight_dims[1], weight_dims[0]});
    layer_resource->weight_handle = weight_handle;

    net_resource.resource_map[layer_resource->name] = std::shared_ptr<TNN_NS::LayerResource>(layer_resource);

    param->num_output = weight_dims[1];
    param->has_bias   = 1;
    param->transpose  = 0;
    param->axis       = 1;
    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_TORCH_OP_CONVERTER(Addmm, aten, addmm);

}  // namespace TNN_CONVERTER
