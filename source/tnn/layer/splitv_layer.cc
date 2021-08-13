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

namespace TNN_NS {

DECLARE_LAYER(SplitV, LAYER_SPLITV);

Status SplitVLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status SplitVLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);
    
    auto layer_param = dynamic_cast<SplitVLayerParam*>(param_);
    if (!layer_param) {
        return Status(TNNERR_PARAM_ERR, "SplitVLayer do not have valid param, please check node: " + layer_name_);
    }
    Blob* input_blob = input_blobs_[0];
    auto input_dims  = input_blob->GetBlobDesc().dims;
    //
    if (layer_param->axis < 0) {
        layer_param->axis += input_dims.size();
    }
    // slices may be empty
    if (layer_param->slices.empty() || !layer_param->is_split_specified) {
        layer_param->slices.clear();
        int input_size       = input_dims[layer_param->axis];
        int output_blob_size = output_blobs_.size();
        if (input_size % output_blob_size == 0) {
            for (int i = 0; i < output_blob_size; ++i) {
                layer_param->slices.push_back(input_size / output_blob_size);
            }
            layer_param->is_split_specified = false;
        } else {
            return Status(
                TNNERR_PARAM_ERR,
                "SplitVLayer has invalid param, slice size is zero, Input cannot be split evenly on select axis");
        }
    }
    if (layer_param->slices.size() != output_blobs_.size()) {
        return Status(TNNERR_PARAM_ERR, "SplitVLayer has invalid param, slices size != output blobs size ");
    }

    int size_sum = layer_param->slices[0];
    for (int i = 1; i < layer_param->slices.size(); i++) {
        size_sum += layer_param->slices[i];
    }

    if (size_sum != input_dims[layer_param->axis]) {
        return Status(TNNERR_PARAM_ERR, "SplitVLayer has invalid slices");
    }

    for (size_t i = 0; i < output_blobs_.size(); i++) {
        auto output_dims                     = input_dims;
        output_dims[layer_param->axis]       = layer_param->slices[i];
        output_blobs_[i]->GetBlobDesc().dims = output_dims;
    }

    return TNN_OK;
}

REGISTER_LAYER(SplitV, LAYER_SPLITV);

}  // namespace TNN_NS
