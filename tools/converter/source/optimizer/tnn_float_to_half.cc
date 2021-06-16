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

#include "tnn/utils/half_utils.h"
#include "tnn_optimize_pass.h"

namespace TNN_CONVERTER {

TNN_NS::RawBuffer FloatToHalf(TNN_NS::RawBuffer src_buffer) {
    int data_size                 = src_buffer.GetBytesSize() / 4;
    TNN_NS::RawBuffer half_buffer = TNN_NS::RawBuffer(data_size * 2);
    TNN_NS::ConvertFromFloatToHalf(src_buffer.force_to<float*>(), half_buffer.force_to<void*>(), data_size);
    half_buffer.SetDataType(TNN_NS::DATA_TYPE_HALF);
    return half_buffer;
}

DECLARE_OPTIMIZE_PASS(FloatToHalf);

std::string TnnOptimizeFloatToHalfPass::PassName() {
    return "FloatToHalf";
}

TNN_NS::Status TnnOptimizeFloatToHalfPass::exec(TNN_NS::NetStructure& net_structure,
                                                TNN_NS::NetResource& net_resource) {
    auto& layers                         = net_structure.layers;
    const std::string conv_output_suffix = "_output";
    const std::string activation_suffix  = "_activation";

    auto& resource_map = net_resource.resource_map;
    int layers_size    = layers.size();
    for (int i = 0; i < layers_size; i++) {
        auto& layer      = layers[i];
        const auto& name = layer->name;
        if (resource_map.find(name) == resource_map.end()) {
            continue;
        }
        if (layer->type == TNN_NS::LAYER_CONVOLUTION) {
            auto* layer_param             = static_cast<TNN_NS::ConvLayerParam*>(layer->param.get());
            auto* layer_resource          = static_cast<TNN_NS::ConvLayerResource*>(resource_map[name].get());
            layer_resource->filter_handle = FloatToHalf(layer_resource->filter_handle);
            if (layer_param->bias) {
                layer_resource->bias_handle = FloatToHalf(layer_resource->bias_handle);
            }
        } else if (layer->type == TNN_NS::LAYER_INNER_PRODUCT) {
            auto* layer_param             = static_cast<TNN_NS::InnerProductLayerParam*>(layer->param.get());
            auto* layer_resource          = static_cast<TNN_NS::InnerProductLayerResource*>(resource_map[name].get());
            layer_resource->weight_handle = FloatToHalf(layer_resource->weight_handle);
            if (layer_param->has_bias) {
                layer_resource->bias_handle = FloatToHalf(layer_resource->bias_handle);
            }
        } else if (layer->type == TNN_NS::LAYER_ELTWISE) {
            auto* layer_resource           = static_cast<TNN_NS::EltwiseLayerResource*>(resource_map[name].get());
            layer_resource->element_handle = FloatToHalf(layer_resource->element_handle);
        }
    }

    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_OPTIMIZE_PASS(FloatToHalf);

}  // namespace TNN_CONVERTER