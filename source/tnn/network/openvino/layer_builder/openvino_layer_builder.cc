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

#include "tnn/network/openvino/layer_builder/openvino_layer_builder.h"

#include <mutex>

#include <ngraph/node.hpp>
#include <ngraph/op/op.hpp>

#include "tnn/core/macro.h"
#include "tnn/core/abstract_device.h"
#include "tnn/extern_wrapper/foreign_blob.h"
#include "tnn/extern_wrapper/foreign_tensor.h"
#include "tnn/device/x86/x86_device.h"
#include "tnn/network/openvino/custom_layer/custom_implementation.h"
#include "tnn/network/openvino/layer_builder/adapter_layer_builder.h"

namespace TNN_NS {

OpenVINOLayerBuilder::OpenVINOLayerBuilder(LayerType type): BaseLayerBuilder(type) {
    _x86_map = X86Device::GetLayerCreatorMap();
    _ov_custom_type = CustomOpenvinoLayerManager::GetCustomLayerTypeSet();
    base_layer_ = CreateLayer(type_);
}

OpenVINOLayerBuilder::~OpenVINOLayerBuilder() {
    if (base_layer_) {
        delete base_layer_;
    }
}

Status OpenVINOLayerBuilder::Init(Context* context, LayerParam* param, LayerResource* resource, std::vector<Blob*>& input_blobs,
                       std::vector<Blob*>& output_blobs, AbstractDevice* device) {
    input_blobs_  = input_blobs;
    output_blobs_ = output_blobs;
    
    param_    = param;
    resource_ = resource;

    if (_x86_map.find(type_) != _x86_map.end() && _ov_custom_type.find(type_) != _ov_custom_type.end()) {
        base_layer_->Init(context, param, resource, input_blobs, output_blobs, device);
    } else {
        base_layer_->Init(context, param, resource, input_blobs, output_blobs, GetDevice(DEVICE_NAIVE));
    }

    RETURN_ON_NEQ(Build(), TNN_OK);
    RETURN_ON_NEQ(InferOutputDataType(), TNN_OK);

    LOGD("InferOutputShape: name:%s %s \n", param->name.c_str(), output_blobs[0]->GetBlobDesc().description().c_str());

    return TNN_OK;

}

std::vector<std::shared_ptr<ngraph::Node>> OpenVINOLayerBuilder::GetInputNodes() {
    std::vector<std::shared_ptr<ngraph::Node>> input_nodes;
    for(auto tensor : GetInputTensors()) {
        auto openvino_tensor = std::dynamic_pointer_cast<OpenvinoTensor>(tensor);
        if (openvino_tensor){
            input_nodes.push_back(openvino_tensor->GetNode());
        } else {
            LOGE("Error: OpenVINOLayerBuilder(%s) got none-openvino input tensor\n", layer_name_.c_str());
            return std::vector<std::shared_ptr<ngraph::Node>>();
        }
    }
    return input_nodes;
}

LayerResource* OpenVINOLayerBuilder::GetResource() {
    return resource_;
}

Status OpenVINOLayerBuilder::SetOutputTensors(ngraph::NodeVector nodes) {
    int index = 0;
    for (auto blob : output_blobs_) {
        auto name = blob->GetBlobDesc().name;
        auto tensor = dynamic_cast<ForeignBlob*>(blob)->GetForeignTensor();
        auto openvino_tensor = std::dynamic_pointer_cast<OpenvinoTensor>(tensor);
        openvino_tensor->SetNode(nodes[index]);
        index++;
    }

    return TNN_OK;
}

std::vector<std::shared_ptr<ngraph::Node>> OpenVINOLayerBuilder::GetOutputNodes() {
    std::vector<std::shared_ptr<ngraph::Node>> output_nodes;
    for(auto tensor : GetOutputTensors()) {
        auto openvino_tensor = std::dynamic_pointer_cast<OpenvinoTensor>(tensor);
        if (openvino_tensor){
            output_nodes.push_back(openvino_tensor->GetNode());
        } else {
            LOGE("Error: OpenVINOLayerBuilder(%s) got none-openvino output tensor\n", layer_name_.c_str());
            return std::vector<std::shared_ptr<ngraph::Node>>();
        }
    }
     return output_nodes;
}

Status OpenVINOLayerBuilder::Reshape(){
    return TNN_OK;
}

Status OpenVINOLayerBuilder::Forward(){
    return TNN_OK;
}

void OpenVINOLayerBuilder::SetConstantResource(ConstantResource* consts) {
    BaseLayer::SetConstantResource(consts);
    this->base_layer_->SetConstantResource(consts);
}

std::map<LayerType, std::shared_ptr<LayerBuilderCreator>>& GetOpenVINOLayerBuilderCreatorMap() {
    // static shared_ptr of LayerCreatorMap.
    static std::once_flag once;
    static std::shared_ptr<std::map<LayerType, std::shared_ptr<LayerBuilderCreator>>> creators;
    std::call_once(once, []() { creators.reset(new std::map<LayerType, std::shared_ptr<LayerBuilderCreator>>); });
    return *creators;
}

OpenVINOLayerBuilder* CreateOpenVINOLayerBuilder(LayerType type) {
    OpenVINOLayerBuilder* cur_layer    = NULL;
    auto& map = GetOpenVINOLayerBuilderCreatorMap();
    if (map.count(type) > 0) {
        auto base_layer = map[type]->CreateLayerBuilder();
        cur_layer = dynamic_cast<OpenVINOLayerBuilder*>(base_layer);
    } else {
        return new AdapterOVLayerBuilder(type);
    }
    return cur_layer;
}

}  // namespace TNN_NS
