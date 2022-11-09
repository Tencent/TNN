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

#include "torch/torch_base_converter.h"
#include "torch/torch_utils.h"

namespace TNN_CONVERTER {

DECLARE_TORCH_OP_CONVERTER(Cat);

std::string TorchCatConverter::TNNOpType(const torch::jit::Node *node, bool quantized_model) {
    return "Concat";
}

TNN_NS::ActivationType TorchCatConverter::ActivationType(const torch::jit::Node *node) {
    return TNN_NS::ActivationType_None;
}

TNN_NS::Status TorchCatConverter::exec(tnn::NetStructure &net_structure, tnn::NetResource &net_resource,
                                       const torch::jit::Node *node, bool quantized_mode) {
    auto cur_layer      = std::make_shared<TNN_NS::LayerInfo>();
    cur_layer->name     = node->output(0)->debugName();
    auto type_name      = TNNOpType(node, quantized_mode);
    auto layer_type     = TNN_NS::GlobalConvertLayerType(type_name);
    cur_layer->type     = layer_type;
    cur_layer->type_str = type_name;
    net_structure.layers.push_back(cur_layer);

    auto param       = new TNN_NS::ConcatLayerParam;
    cur_layer->param = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->type      = cur_layer->type_str;
    param->name      = cur_layer->name;
    param->quantized = false;
    param->axis      = (int)GetValue<int64_t>(node->input(1));

    const auto &tensor_list = node->input(0);
    const auto &inputs      = tensor_list->node()->inputs();
    for (const auto &input : inputs) {
        cur_layer->inputs.push_back(input->debugName());
    }
    for (const auto &input : inputs) {
        if (!toIValue(input)) {
            continue;
        }
        auto const_buf = CreateRawBufferFromValue(input);
        if (const_buf.GetBytesSize() > 0) {
            if (*(const_buf.force_to<int *>()) != INT_MAX) {
                const_buf.SetBufferDims({1});
                net_resource.constant_map[input->debugName()] = std::make_shared<TNN_NS::RawBuffer>(const_buf);
            }
        }
    }
    cur_layer->outputs.push_back(node->output(0)->debugName());
    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_TORCH_OP_CONVERTER(Cat, aten, cat);

}  // namespace TNN_CONVERTER
