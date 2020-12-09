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
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/data_flag_utils.h"

namespace TNN_NS {
DECLARE_LAYER(Unsqueeze, LAYER_UNSQUEEZE);

Status UnsqueezeLayer::InferOutputDataType() {
    //return BaseLayer::InferOutputDataType();
    BaseLayer::InferOutputDataType();

    int flag = DATA_FLAG_CHANGE_NEVER;
    for (auto iter : input_blobs_) {
        if (const_resource_.find(iter->GetBlobDesc().name) != const_resource_.end()) {
            flag |= DATA_FLAG_CHANGE_IF_SHAPE_DIFFER;
        }
    }
        
    for (auto& iter : output_blobs_) {
        if(flag & DATA_FLAG_CHANGE_IF_SHAPE_DIFFER) {
            iter->flag = DATA_FLAG_CHANGE_IF_SHAPE_DIFFER;
        }
    }
    return TNN_OK;
}

Status UnsqueezeLayer::InferOutputShape() {
    BaseLayer::InferOutputShape();
    
    //ASSERT(input_blobs_.size() == 1);
    auto layer_param      = dynamic_cast<UnsqueezeLayerParam*>(param_);
    auto layer_resource = dynamic_cast<UnsqueezeLayerResource*>(resource_);
    
    const auto& output_blob = output_blobs_[0];
    
    DimsVector output_dims;
    if (layer_param->data_in_resource) {
        output_dims = layer_resource->data_dims;
    } else {
        output_dims = input_blobs_[0]->GetBlobDesc().dims;
    }
    
    auto axes = layer_param->axes;
    for (auto axis : axes) {
        if (axis < 0) {
            axis += output_dims.size();
        }
        output_dims.insert(output_dims.begin() + axis, 1);
    }
    
    output_blob->GetBlobDesc().dims = output_dims;
    
    return TNN_OK;
}

REGISTER_LAYER(Unsqueeze, LAYER_UNSQUEEZE);

}  // namespace TNN_NS
