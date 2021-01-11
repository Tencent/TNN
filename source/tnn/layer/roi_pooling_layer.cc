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

namespace TNN_NS {

DECLARE_LAYER(RoiPooling, LAYER_ROIPOOLING);

Status RoiPoolingLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status RoiPoolingLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);
    
    Blob* input_blob = input_blobs_[0];
    Blob* rpn_blob   = input_blobs_[1];  // of shape <n, num_rois, 5, 1 ,1>

    RoiPoolingLayerParam* pool_param = dynamic_cast<RoiPoolingLayerParam*>(param_);

    bool is_5d_input = input_blob->GetBlobDesc().dims.size() == 5;

    int idx      = 0;
    int num      = input_blob->GetBlobDesc().dims[idx++];
    int channels = input_blob->GetBlobDesc().dims[idx++];
    int depth    = is_5d_input ? input_blob->GetBlobDesc().dims[idx++] : 1;
    int height   = input_blob->GetBlobDesc().dims[idx++];
    int width    = input_blob->GetBlobDesc().dims[idx++];

    DimsVector output_dims;
    output_dims.push_back(num);
    output_dims.push_back(rpn_blob->GetBlobDesc().dims[1]);
    if (is_5d_input) {
        output_dims.push_back(pool_param->pooled_dims[2]);
    }
    output_dims.push_back(pool_param->pooled_dims[1]);
    output_dims.push_back(pool_param->pooled_dims[0]);

    for (int i = 0; i < output_blobs_.size(); ++i) {
        output_blobs_[i]->GetBlobDesc().dims = output_dims;
    }
    return TNN_OK;
}

REGISTER_LAYER(RoiPooling, LAYER_ROIPOOLING);

}  // namespace TNN_NS
