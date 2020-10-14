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

#include <cmath>

#include "tnn/layer/base_layer.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

DECLARE_LAYER(Expand, LAYER_EXPAND);

Status ExpandLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status ExpandLayer::InferOutputShape() {
    auto expand_param = dynamic_cast<ExpandLayerParam*>(param_);
    CHECK_PARAM_NULL(expand_param);
    Blob* input_blob = input_blobs_[0];
    Blob* output_blob = output_blobs_[0];
    auto input_dims = input_blob->GetBlobDesc().dims;
    auto shape_dims = expand_param->shape;
    DimsVector max_dims, min_dims;
    if(input_dims.size() >= shape_dims.size()) {
        max_dims = input_dims;
        min_dims = shape_dims;
    } else {
        max_dims = shape_dims;
        min_dims = input_dims;
    }
    auto output_dims = max_dims;
    int diff = max_dims.size() - min_dims.size();
    for(int i = 0; i < min_dims.size(); ++i) {
        if(min_dims[i] != 1 && min_dims[i] != max_dims[diff + i]) {
            if(max_dims[diff + i] == 1) {
                output_dims[diff + i] = min_dims[i];
            } else {
                return Status(TNNERR_PARAM_ERR, "expand param dims error"); 
            }
        }
    } 
    output_blob->GetBlobDesc().dims = output_dims;
    return TNN_OK;
}

REGISTER_LAYER(Expand, LAYER_EXPAND);

}  // namespace TNN_NS
