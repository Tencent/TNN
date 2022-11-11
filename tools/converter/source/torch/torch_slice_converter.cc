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

DECLARE_TORCH_OP_CONVERTER(Slice);

std::string TorchSliceConverter::TNNOpType(const torch::jit::Node *node, bool quantized_model) {
    return "StridedSliceV2";
}

TNN_NS::ActivationType TorchSliceConverter::ActivationType(const torch::jit::Node *node) {
    return TNN_NS::ActivationType_None;
}

static int Int64ToInt32(const int64_t number) {
    if (number < INT_MIN) {
        return INT_MIN;
    } else if (number > INT_MAX) {
        return INT_MAX;
    }

    return (int)number;
}
// "slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)"
TNN_NS::Status TorchSliceConverter::exec(tnn::NetStructure &net_structure, tnn::NetResource &net_resource,
                                         const torch::jit::Node *node, bool quantized_mode) {
    auto cur_layer      = std::make_shared<TNN_NS::LayerInfo>();
    cur_layer->name     = node->output(0)->debugName();
    auto type_name      = TNNOpType(node, quantized_mode);
    auto layer_type     = TNN_NS::GlobalConvertLayerType(type_name);
    cur_layer->type     = layer_type;
    cur_layer->type_str = type_name;
    net_structure.layers.push_back(cur_layer);

    auto param       = new TNN_NS::StrideSliceV2LayerParam;
    cur_layer->param = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->type      = cur_layer->type_str;
    param->name      = cur_layer->name;
    param->quantized = false;

    const auto inputs = node->inputs();
    if (inputs.size() != 5) {
        return TNN_NS::TNNERR_CONVERT_UNSUPPORT_LAYER;
    }
    const auto &axis = inputs[1];
    if (toIValue(axis)) {
        param->axes.push_back(Int64ToInt32(GetValue<int64_t>(axis)));
        TNN_NS::RawBuffer const_buf                 = CreateRawBufferFromValue(axis);
        net_resource.constant_map[axis->debugName()] = std::make_shared<TNN_NS::RawBuffer>(const_buf);
    }
    const auto &start = inputs[2];
    if (toIValue(start)) {
        param->begins.push_back(Int64ToInt32(GetValue<int64_t>(start)));
        TNN_NS::RawBuffer const_buf                   = CreateRawBufferFromValue(start);
        net_resource.constant_map[start->debugName()] = std::make_shared<TNN_NS::RawBuffer>(const_buf);
    }
    const auto &end = inputs[3];
    if (toIValue(end)) {
        param->ends.push_back(Int64ToInt32(GetValue<int64_t>(end)));
        TNN_NS::RawBuffer const_buf                 = CreateRawBufferFromValue(end);
        net_resource.constant_map[end->debugName()] = std::make_shared<TNN_NS::RawBuffer>(const_buf);
    }
    const auto &step = inputs[4];
    if (toIValue(step)) {
        param->strides.push_back(Int64ToInt32(GetValue<int64_t>(step)));
        TNN_NS::RawBuffer const_buf                 = CreateRawBufferFromValue(step);
        net_resource.constant_map[step->debugName()] = std::make_shared<TNN_NS::RawBuffer>(const_buf);
    }
    // Tensor = aten::slice(%x.2, %3, %4, %14, %6)
    cur_layer->inputs.push_back(inputs[0]->debugName());
    cur_layer->inputs.push_back(start->debugName());
    cur_layer->inputs.push_back(end->debugName());
    cur_layer->inputs.push_back(axis->debugName());
    cur_layer->inputs.push_back(step->debugName());

    cur_layer->outputs.push_back(node->output(0)->debugName());

    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_TORCH_OP_CONVERTER(Slice, aten, slice);

}  // namespace TNN_CONVERTER
