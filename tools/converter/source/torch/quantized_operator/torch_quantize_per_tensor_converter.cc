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

#include "tnn/interpreter/tnn/objseri.h"
#include "tnn/utils/dims_utils.h"
#include "torch/torch_base_converter.h"
#include "torch/torch_utils.h"

namespace TNN_CONVERTER {

DECLARE_TORCH_OP_CONVERTER(Quantize);

std::string TorchQuantizeConverter::TNNOpType(const torch::jit::Node *node, bool quantized_model) {
    return "Int8Quantized";
}

TNN_NS::ActivationType TorchQuantizeConverter::ActivationType(const torch::jit::Node *node) {
    return TNN_NS::ActivationType_None;
}

TNN_NS::Status TorchQuantizeConverter::exec(tnn::NetStructure &net_structure, tnn::NetResource &net_resource,
                                            const torch::jit::Node *node, bool quantized_mode) {
    auto cur_layer          = std::make_shared<TNN_NS::LayerInfo>();
    cur_layer->name         = node->output(0)->debugName();
    auto type_name          = TNNOpType(node, quantized_mode);
    auto layer_type         = TNN_NS::GlobalConvertLayerType(type_name);
    cur_layer->type         = layer_type;
    cur_layer->type_str     = type_name;
    std::string input_name  = node->input(0)->debugName();
    std::string outptu_name = node->output(0)->debugName();
    cur_layer->inputs.push_back(input_name);
    cur_layer->outputs.push_back(node->output(0)->debugName());
    net_structure.layers.push_back(cur_layer);

    auto param       = new TNN_NS::LayerParam;
    cur_layer->param = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->type      = cur_layer->type_str;
    param->name      = cur_layer->name;
    param->quantized = false;

    // create output blob scale
    std::string output_blob_scale_name = outptu_name + BLOB_SCALE_SUFFIX;
    auto &resource_map                 = net_resource.resource_map;
    if (resource_map.find(output_blob_scale_name) == resource_map.end()) {
        auto *output_blob_scale                            = new TNN_NS::IntScaleResource;
        output_blob_scale->name                            = output_blob_scale_name;
        output_blob_scale->scale_handle                    = CreateRawBufferFromValue(node->input(1));
        TNN_NS::RawBuffer zero_point_handle                = CreateRawBufferFromValue(node->input(2));
        output_blob_scale->zero_point_handle               = ConvertRawBufferToZero(zero_point_handle);
        net_resource.resource_map[output_blob_scale->name] = std::shared_ptr<TNN_NS::LayerResource>(output_blob_scale);
    }
    // create extra input blob scale, because EliminateUnusefulNode pass
    std::string input_blob_scale_name = input_name + BLOB_SCALE_SUFFIX;
    if (resource_map.find(input_blob_scale_name) == resource_map.end()) {
        auto *input_blob_scale                            = new TNN_NS::IntScaleResource;
        input_blob_scale->name                            = input_blob_scale_name;
        input_blob_scale->scale_handle                    = CreateRawBufferFromValue(node->input(1));
        TNN_NS::RawBuffer input_zero_point_handle         = CreateRawBufferFromValue(node->input(2));
        input_blob_scale->zero_point_handle               = ConvertRawBufferToZero(input_zero_point_handle);
        net_resource.resource_map[input_blob_scale->name] = std::shared_ptr<TNN_NS::LayerResource>(input_blob_scale);
    }

    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_TORCH_OP_CONVERTER(Quantize, aten, quantize_per_tensor);

}  // namespace TNN_CONVERTER