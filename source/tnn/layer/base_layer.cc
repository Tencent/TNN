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

#include "tnn/layer/base_layer.h"

#include <mutex>

#include "tnn/core/macro.h"

namespace TNN_NS {
BaseLayer::BaseLayer(LayerType type) {
    this->type_      = type;
    this->layer_acc_ = nullptr;
    this->param_     = nullptr;
    this->resource_  = nullptr;
}

BaseLayer::~BaseLayer() {
    if (layer_acc_ != NULL) {
        delete layer_acc_;
        layer_acc_ = NULL;
    }
}

Status BaseLayer::Init(Context* context, LayerParam* param, LayerResource* resource, std::vector<Blob*>& input_blobs,
                       std::vector<Blob*>& output_blobs, AbstractDevice* device) {
    input_blobs_  = input_blobs;
    output_blobs_ = output_blobs;

    param_    = param;
    resource_ = resource;

    auto status = InferOutputDataType();
    if (status != TNN_OK) {
        return status;
    }

    status = InferOutputShape();
    if (status != TNN_OK) {
        return status;
    }
    for (auto& output_blob : output_blobs) {
        LOGD("InferOutputShape: name:%s shape:%d %d %d %d \n", output_blob->GetBlobDesc().name.c_str(),
             output_blob->GetBlobDesc().dims[0], output_blob->GetBlobDesc().dims[1], output_blob->GetBlobDesc().dims[2],
             output_blob->GetBlobDesc().dims[3]);
    }
    auto dims = output_blobs[0]->GetBlobDesc().dims;
    for (auto item : dims) {
        if (item <= 0) {
            LOGE("Error: layer(%s) output dims is invalid\n", layer_name_.c_str());
            return Status(TNNERR_LAYER_ERR, "layer output dims is invalid");
        }
    }

    layer_acc_ = device->CreateLayerAcc(type_);
    if (layer_acc_ != NULL) {
        return layer_acc_->Init(context, param, resource, input_blobs_, output_blobs_);
    } else {
        LOGE("layer acc of type(%d) is nil\n", type_);
        return Status(TNNERR_LAYER_ERR, "layer acc is nil");
    }
}

Status BaseLayer::InferOutputDataType() {
    // Init base type, will re write in different device acc
    // output data_type = input_data_tyep as default.
    for (auto output_blob : output_blobs_) {
        output_blob->GetBlobDesc().data_type = input_blobs_[0]->GetBlobDesc().data_type;
    }
    return TNN_OK;
}

Status BaseLayer::Reshape() {
    InferOutputShape();
    auto dims = output_blobs_[0]->GetBlobDesc().dims;
    for (auto item : dims) {
        if (item <= 0) {
            LOGE("Error: layer(%s) output dims is invalid\n", layer_name_.c_str());
            return Status(TNNERR_LAYER_ERR, "layer output dims is invalid");
        }
    }

    if (layer_acc_ != NULL) {
        return layer_acc_->Reshape(input_blobs_, output_blobs_);
    } else {
        LOGE("layer acc is nil\n");
        return Status(TNNERR_LAYER_ERR, "layer acc is nil");
    }
}

Status BaseLayer::Forward() {
    if (layer_acc_ != NULL) {
        return layer_acc_->Forward(input_blobs_, output_blobs_);
    } else {
        LOGE("layer acc is nil\n");
        return Status(TNNERR_LAYER_ERR, "layer acc is nil");
    }
}

void BaseLayer::SetLayerName(std::string layer_name) {
    layer_name_ = layer_name;
}

std::string BaseLayer::GetLayerName() {
    return layer_name_;
}

//@brief get all input blobs
std::vector<Blob*> BaseLayer::GetInputBlobs() {
    return input_blobs_;
}

//@brief get all input blobs
std::vector<Blob*> BaseLayer::GetOutputBlobs() {
    return output_blobs_;
}

Status BaseLayer::InferShapeAhead(std::vector<Blob*>& input_blobs, std::vector<Blob*>& output_blobs, LayerParam* param,
                                  LayerResource* resource) {
    input_blobs_  = input_blobs;
    output_blobs_ = output_blobs;
    param_        = param;
    resource_     = resource;

    InferOutputShape();
    return TNN_OK;
}

std::map<LayerType, std::shared_ptr<LayerCreator>>& GetGlobalLayerCreatorMap() {
    // static shared_ptr of LayerCreatorMap.
    static std::once_flag once;
    static std::shared_ptr<std::map<LayerType, std::shared_ptr<LayerCreator>>> creators;
    std::call_once(once, []() { creators.reset(new std::map<LayerType, std::shared_ptr<LayerCreator>>); });
    return *creators;
}

BaseLayer* CreateLayer(LayerType type) {
    BaseLayer* cur_layer    = NULL;
    auto& layer_creater_map = GetGlobalLayerCreatorMap();
    if (layer_creater_map.count(type) > 0) {
        cur_layer = layer_creater_map[type]->CreateLayer();
    }
    return cur_layer;
}

}  // namespace TNN_NS
