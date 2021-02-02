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
#include "tnn/utils/data_flag_utils.h"
#include "tnn/utils/string_utils_inner.h"

#include <mutex>
#include <sstream>

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
        LOGE("InferOutputDataType failed\n");
        return status;
    }
    
    if (!output_blobs_[0]->NeedAllocateInForward()){
        status = InferOutputShape();
        if (status != TNN_OK) {
            LOGE("InferOutputShape failed\n");
            return status;
        }
    }
    
    if (runtime_model_ == RUNTIME_MODE_NORMAL) {
        for (auto& output_blob : output_blobs) {
            LOGD("InferOutputShape: %s\n", output_blob->GetBlobDesc().description().c_str());
        }
        auto dims = output_blobs[0]->GetBlobDesc().dims;
        for (auto item : dims) {
            if (item <= 0) {
                LOGE("Error: layer(%s) output dims is invalid\n", layer_name_.c_str());
                return Status(TNNERR_LAYER_ERR, "layer output dims is invalid");
            }
        }
    }
    
    if (device->GetDeviceType() == DEVICE_NAIVE || !IsOutputConstant()) {
        layer_acc_ = device->CreateLayerAcc(type_);
        if (layer_acc_ != NULL) {
            layer_acc_->SetRuntimeMode(runtime_model_);
            layer_acc_->SetConstantResource(const_resource_);
            return layer_acc_->Init(context, param, resource, input_blobs_, output_blobs_);
        } else {
            LOGE("layer acc of type(%d) is nil\n", type_);
            return Status(TNNERR_LAYER_ERR, "layer acc is nil");
        }
    }
    return TNN_OK;
}

Status BaseLayer::FillLayerParamWithConstantResource() {
    return TNN_OK;
}

Status BaseLayer::InferOutputShape(bool ignore_error) {
    //get dims from const for input
    auto const_resource = const_resource_;
    for (auto iter : input_blobs_) {
        auto name = iter->GetBlobDesc().name;
        if (const_resource == nullptr || const_resource->find(name) == const_resource->end()) {
            continue;
        }
        iter->GetBlobDesc().data_type = (*const_resource)[name]->GetDataType();
        
        //only DATA_FLAG_CHANGE_NEVER read dims and type from const resource
        //blob with flag DATA_FLAG_CHANGE_IF_SHAPE_DIFFER may change dims in runtime
        if (DataFlagUtils::ChangeStatus(iter->flag) == DATA_FLAG_CHANGE_NEVER) {
            iter->GetBlobDesc().dims = (*const_resource)[name]->GetBufferDims();
        }
    }
    
    //
    if (runtime_model_ == RUNTIME_MODE_NORMAL || GetLayerChangeFlag() == DATA_FLAG_CHANGE_NEVER) {
        return FillLayerParamWithConstantResource();
    }
    return TNN_OK;
}

Status BaseLayer::InferOutputDataType() {
    auto const_resource = const_resource_;
    
    // Init base type, will re write in different device acc
    // output data_type = input_data_tyep as default.
    
    //find first blob which is not const
    auto input_blob_not_const = input_blobs_[0];
    for (auto input_blob : input_blobs_) {
        if (const_resource == nullptr || const_resource->find(input_blob->GetBlobDesc().name) == const_resource->end()) {
            input_blob_not_const = input_blob;
            break;
        }
    }
    
    for (auto output_blob : output_blobs_) {
        output_blob->GetBlobDesc().data_type = input_blob_not_const->GetBlobDesc().data_type;
    }
    
    int flag = DATA_FLAG_CHANGE_NEVER;
    for (auto iter : input_blobs_) {
        if (const_resource != nullptr && const_resource->find(iter->GetBlobDesc().name) != const_resource->end()) {
            iter->flag |= DATA_FLAG_CHANGE_NEVER;
        }
        flag = DataFlagUtils::MinChangeStatus(flag, iter->flag);
    }
    
    for (auto iter : output_blobs_) {
        if (runtime_model_ == RUNTIME_MODE_NORMAL) {
            if (const_resource != nullptr && const_resource->find(iter->GetBlobDesc().name) != const_resource->end()) {
                flag = flag & 0x0000FFFF;
            }
        } else {
            //allocate output blob of const layer in const folding
            if (DataFlagUtils::ChangeStatus(flag) != DATA_FLAG_CHANGE_ALWAYS) {
                flag = flag | DATA_FLAG_ALLOCATE_IN_FORWARD;
            }
        }

        iter->flag = flag;
    }
    return TNN_OK;
}

Status BaseLayer::Reshape() {
    if (!output_blobs_[0]->NeedAllocateInForward()){
        auto status = InferOutputShape();
        RETURN_ON_NEQ(status, TNN_OK);
        
        auto dims = output_blobs_[0]->GetBlobDesc().dims;
        for (auto item : dims) {
            if (item < 0) {
                LOGE("Error: layer(%s) output dims is invalid\n", layer_name_.c_str());
                return Status(TNNERR_LAYER_ERR, "layer output dims is invalid");
            }
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
        if (runtime_model_ == RUNTIME_MODE_NORMAL) {
            auto status = layer_acc_->BeforeForward(input_blobs_, output_blobs_);
            RETURN_ON_NEQ(status, TNN_OK);
            
            if (!IsOutputConstant()) {
                status = layer_acc_->Forward(input_blobs_, output_blobs_);
                RETURN_ON_NEQ(status, TNN_OK);
            }
        } else {
            //dont check the status of InferOutputShape in constant folding
            auto status = InferOutputShape(true);
            
            status = layer_acc_->BeforeForward(input_blobs_, output_blobs_);
            RETURN_ON_NEQ(status, TNN_OK);
            
            if (IsOutputConstant()) {
                status = layer_acc_->Forward(input_blobs_, output_blobs_);
                RETURN_ON_NEQ(status, TNN_OK);
            }
        }
        
        return layer_acc_->AfterForward(input_blobs_, output_blobs_);
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

void BaseLayer::SetRuntimeBlobMemoryPool(BlobMemoryPool *runtime_blob_pool) {
    if (layer_acc_) {
        layer_acc_->SetRuntimeBlobMemoryPool(runtime_blob_pool);
    }
}

bool BaseLayer::IsOutputConstant() {
    for (auto iter : output_blobs_) {
        if (!iter->IsConstant()) {
            return false;
        }
    }
    return true;
}

int BaseLayer::GetLayerChangeFlag() {
    int flag = DATA_FLAG_CHANGE_NEVER;
    for (auto iter : output_blobs_) {
        flag = DataFlagUtils::ChangeStatus(DataFlagUtils::MinChangeStatus(flag, iter->flag));
    }
    return flag;
}

void BaseLayer::SetConstantResource(ConstantResource* consts) {
    const_resource_ = consts;
}

// @brief set runtime mode
void BaseLayer::SetRuntimeMode(RuntimeMode mode) {
    runtime_model_ = mode;
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
