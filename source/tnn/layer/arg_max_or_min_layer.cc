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
DECLARE_LAYER(ArgMaxOrMin, LAYER_ARG_MAX_OR_MIN);

Status ArgMaxOrMinLayer::InferOutputDataType() {
    BaseLayer::InferOutputDataType();
    output_blobs_[0]->GetBlobDesc().data_type = DATA_TYPE_INT32;
    
    return TNN_OK;
}

Status ArgMaxOrMinLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);
    
    auto param                      = dynamic_cast<ArgMaxOrMinLayerParam*>(param_);
    CHECK_PARAM_NULL(param);
    auto input_blob                 = input_blobs_[0];
    auto output_blob                = output_blobs_[0];
    auto output_dims = input_blob->GetBlobDesc().dims;
    if (param->axis < 0) {
        param->axis += input_blob->GetBlobDesc().dims.size();
    }
    
    if (!param->keep_dims) {
        output_dims.erase(output_dims.begin() + param->axis);
    } else {
        output_dims[param->axis] = 1;
    }
    
    output_blob->GetBlobDesc().dims = output_dims;
    return TNN_OK;
}

REGISTER_LAYER(ArgMaxOrMin, LAYER_ARG_MAX_OR_MIN);

}  // namespace TNN_NS
