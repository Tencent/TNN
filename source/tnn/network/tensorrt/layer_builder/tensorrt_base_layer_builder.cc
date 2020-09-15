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

#include <mutex>

#include "tnn/network/tensorrt/layer_builder/tensorrt_base_layer_builder.h"

namespace TNN_NS {

TensorRTBaseLayerBuilder::TensorRTBaseLayerBuilder(LayerType type) : BaseLayerBuilder(type), trt_batchsize(0) {
    m_layer = CreateLayer(type);
}

TensorRTBaseLayerBuilder::~TensorRTBaseLayerBuilder() {
}

Status TensorRTBaseLayerBuilder::InferOutputShape() {
    return TNN_OK;
}

Status TensorRTBaseLayerBuilder::Build() {
    return TNN_OK;
}

bool TensorRTBaseLayerBuilder::IsPluginLayer() {
    return this->is_plugin;
}

void TensorRTBaseLayerBuilder::SetBatchSize(int value) {
    this->trt_batchsize = value;
}

std::map<LayerType, std::shared_ptr<LayerBuilderCreator>>& GetTensorRTLayerBuilderCreatorMap() {
    // static shared_ptr of LayerCreatorMap.
    static std::once_flag once;
    static std::shared_ptr<std::map<LayerType, std::shared_ptr<LayerBuilderCreator>>> creators;
    std::call_once(once, []() { creators.reset(new std::map<LayerType, std::shared_ptr<LayerBuilderCreator>>); });
    return *creators;
}

std::map<LayerType, std::shared_ptr<LayerBuilderCreator>>& GetTensorRTPluginLayerBuilderCreatorMap() {
    // static shared_ptr of LayerCreatorMap.
    static std::once_flag once;
    static std::shared_ptr<std::map<LayerType, std::shared_ptr<LayerBuilderCreator>>> creators;
    std::call_once(once, []() { creators.reset(new std::map<LayerType, std::shared_ptr<LayerBuilderCreator>>); });
    return *creators;
}

TensorRTBaseLayerBuilder* CreateTensorRTBaseLayerBuilder(LayerType type) {
    TensorRTBaseLayerBuilder* cur_layer = nullptr;
    auto& trt_map = GetTensorRTLayerBuilderCreatorMap();
    auto& plugin_map = GetTensorRTPluginLayerBuilderCreatorMap();
    if (trt_map.count(type) > 0) {
        auto base_layer = trt_map[type]->CreateLayerBuilder();
        cur_layer = dynamic_cast<TensorRTBaseLayerBuilder*>(base_layer);
    } else if (plugin_map.count(type) > 0) {
        auto base_layer = plugin_map[type]->CreateLayerBuilder();
        cur_layer = dynamic_cast<TensorRTBaseLayerBuilder*>(base_layer);
    }
    return cur_layer;
}

}  //  namespace TNN_NS