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

#include "base_layer.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_LAYER_WITH_FUNC(OneHot, LAYER_ONEHOT,
                        virtual Status FillLayerParamWithConstantResource(););

Status OneHotLayer::InferOutputDataType() {
    BaseLayer::InferOutputDataType();
    
    DataType output_data_type = DATA_TYPE_INT32;
    if (input_blobs_.size() >=3) {
        output_data_type = input_blobs_[2]->GetBlobDesc().data_type;
    }
    
    for (auto output_blob : output_blobs_) {
        output_blob->GetBlobDesc().data_type = output_data_type;
    }
    return TNN_OK;
}

Status OneHotLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);
    
    auto layer_param = dynamic_cast<OneHotLayerParam*>(param_);
    RETURN_VALUE_ON_NEQ(!layer_param, false,
                        Status(TNNERR_PARAM_ERR,"OneHotLayerParam is nil"));
    
    int axis = layer_param->axis;
    auto output_dims = input_blobs_[0]->GetBlobDesc().dims;
    if (axis < 0) {
        axis += output_dims.size() + 1;
    }
    
    output_dims.insert(output_dims.begin()+axis, layer_param->depth);

    output_blobs_[0]->GetBlobDesc().dims = output_dims;
    return TNN_OK;
}

Status OneHotLayer::FillLayerParamWithConstantResource() {
    Status status = TNN_OK;
    auto *layer_param = dynamic_cast<OneHotLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    
    if (input_blobs_.size() < 3) {
        return Status(TNNERR_PARAM_ERR, "OneHotLayer has invalid layer param");
    }
    
    //depth
    {
        const auto res_name = input_blobs_[1]->GetBlobDesc().name;
        if (const_resource_ != nullptr && const_resource_->find(res_name) != const_resource_->end()) {
            auto buffer = (*const_resource_)[res_name];
            if (buffer->GetDataType() != DATA_TYPE_INT32) {
                return Status(TNNERR_PARAM_ERR, "OneHotLayer has invalid layer resource for depth param");
            }
            
            auto data = buffer->force_to<int *>();
            layer_param->depth = data[0];
        }
    }
    
    //values
    {
        const auto res_name = input_blobs_[1]->GetBlobDesc().name;
        if (const_resource_ != nullptr && const_resource_->find(res_name) != const_resource_->end()) {
            auto buffer = (*const_resource_)[res_name];
            if (buffer->GetDataType() != DATA_TYPE_FLOAT || buffer->GetDataCount() < 2) {
                return Status(TNNERR_PARAM_ERR, "OneHotLayer has invalid layer resource for values param");
            }
            auto data = buffer->force_to<float *>();
            
            layer_param->value_off = data[0];
            layer_param->value_on = data[1];
        }
    }
    return status;
}

REGISTER_LAYER(OneHot, LAYER_ONEHOT);

}  // namespace TNN_NS
