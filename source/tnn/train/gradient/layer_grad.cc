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

#include "tnn/train/gradient/layer_grad.h"

namespace TNN_NS {

LayerGrad::LayerGrad() {}

LayerGrad::~LayerGrad() {}

Status LayerGrad::RegisterLayerGrad(DeviceType device, LayerType type, std::shared_ptr<LayerGrad> layer_grad) {
    GetLayerGradMap()[{device, type}] = layer_grad;
    return TNN_OK;
};

LayerGrad *LayerGrad::GetLayerGrad(DeviceType device, LayerType type) {
    auto &layer_grad_map = GetLayerGradMap();
    if (layer_grad_map.count({device, type}) > 0) {
        return layer_grad_map[{device, type}].get();
    }
    return nullptr;
}

std::map<std::pair<DeviceType, LayerType>, std::shared_ptr<LayerGrad>> &LayerGrad::GetLayerGradMap() {
    static std::map<std::pair<DeviceType, LayerType>, std::shared_ptr<LayerGrad>> layer_grad_map;
    return layer_grad_map;
}

}  // namespace TNN_NS
