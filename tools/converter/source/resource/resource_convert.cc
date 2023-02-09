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

#include "tools/converter/source/resource/resource_convert.h"

#include "tnn/core/common.h"
#include "tools/converter/source/resource/reource_base_convert.h"

namespace TNN_CONVERTER {

TNN_NS::Status ResourceConvert::SetResourceConvertType(ResourceConvertType resource_convert_type) {
    this->resource_convert_type_ = resource_convert_type;
    return TNN_NS::TNN_OK;
}
TNN_NS::Status ResourceConvert::converter(TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource) {
    if (resource_convert_type_ == RESOURCE_KEEP_ORIGINAL) {
        return TNN_NS::TNN_OK;
    }
    auto& resource_map = net_resource.resource_map;
    // convert float weight to half weight
    if (resource_convert_type_ == RESOURCE_CONVERT_HALF) {
        if (net_structure.layers.empty()) {
            return TNN_NS::TNN_OK;
        }
        for (auto& layer : net_structure.layers) {
            const std::string& layer_name = layer->name;
            if (resource_map.find(layer_name) != resource_map.end() &&
                resource_map.find(layer_name)->second != nullptr) {
                const auto& convert = ResourceConvertManager::get()->search(layer->type_str);
                if (convert == nullptr) {
                    LOGE("The ResourceConverter do not support layer:%s \n", layer->name.c_str());
                    LOGE("The unsupported operator type is:%s\n", layer->type_str.c_str());
                    return TNN_NS::TNNERR_CONVERT_UNSUPPORT_LAYER;
                }
                std::shared_ptr<TNN_NS::LayerResource> layer_resource = resource_map.find(layer_name)->second;
                auto status = convert->ConvertToHalfResource(layer->param, layer_resource);
                if (status != TNN_NS::TNN_CONVERT_OK) {
                    LOGE("ResourceConvert failed for %s\n", layer->name.c_str());
                    return status;
                }
            }
        }
    }
    return TNN_NS::TNN_OK;
}
}  // namespace TNN_CONVERTER
