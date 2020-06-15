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

#include "tnn/device/openvino/layer_builder/openvino_layer_builder.h"

#include <mutex>

#include <ngraph/node.hpp>
#include <ngraph/op/op.hpp>

#include "tnn/core/macro.h"
#include "tnn/extern_wrapper/foreign_blob.h"
#include "tnn/extern_wrapper/foreign_tensor.h"
#include "tnn/device/openvino/openvino_types.h"

namespace TNN_NS {

OpenVINOLayerBuilder::OpenVINOLayerBuilder(LayerType type): BaseLayerBuilder(type) {
}

OpenVINOLayerBuilder::~OpenVINOLayerBuilder() {
}

Status OpenVINOLayerBuilder::Init(Context* context, LayerParam* param, LayerResource* resource, std::vector<Blob*>& input_blobs,
                       std::vector<Blob*>& output_blobs, AbstractDevice* device) {

    input_blobs_  = input_blobs;
    output_blobs_ = output_blobs;

    param_    = param;
    resource_ = resource;

    Build();
    SetOutputTensors(output_blobs);

    auto status = InferOutputDataType();
    if (status != TNN_OK) {
        return status;
    }
    
    status = InferOutputShape();
    
    LOGD("InferOutputShape: name:%s shape:%d %d %d %d \n", param->name.c_str(), output_blobs[0]->GetBlobDesc().dims[0],
         output_blobs[0]->GetBlobDesc().dims[1], output_blobs[0]->GetBlobDesc().dims[2],
         output_blobs[0]->GetBlobDesc().dims[3]);
    if (status != TNN_OK) {
        return status;
    }
    
    // auto dims = output_blobs[0]->GetBlobDesc().dims;
    
    // for (auto item : dims) {
    //     if (item <= 0) {
    //         LOGE("Error: layer(%s) output dims is invalid\n", layer_name_.c_str());
    //         return Status(TNNERR_LAYER_ERR, "layer output dims is invalid");
    //     }
    // }

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
    // return inputNodes_;
}

LayerResource* OpenVINOLayerBuilder::GetResource() {
    return resource_;
}

Status OpenVINOLayerBuilder::SetOutputTensors(std::vector<Blob*> output_blobs) {
    for (auto blob : output_blobs) {
        auto name = blob->GetBlobDesc().name;
        auto tensor = dynamic_cast<ForeignBlob*>(blob)->GetForeignTensor();
        auto openvino_tensor = std::dynamic_pointer_cast<OpenvinoTensor>(tensor);
        openvino_tensor->SetNode(outputNodes_[0]);
        for (auto dim : outputNodes_[0]->get_output_shape(0)) {
            blob->GetBlobDesc().dims.push_back(dim);
        }
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
    // return outputNodes_;
}

Status OpenVINOLayerBuilder::SetOutputNodes(ngraph::NodeVector node) {
    outputNodes_ = node;
    return TNN_OK;
}

Status OpenVINOLayerBuilder::Reshape(){
    return TNN_OK;
}

Status OpenVINOLayerBuilder::Forward(){
    return TNN_OK;
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
    }
    return cur_layer;
}

}  // namespace TNN_NS
