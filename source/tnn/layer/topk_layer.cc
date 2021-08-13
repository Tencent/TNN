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
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

//different implementation may generate different order, cause unalign [onnx vs cpu vs other devices]  

DECLARE_LAYER(TopK, LAYER_TOPK);

Status TopKLayer::InferOutputDataType() {
    Status status = BaseLayer::InferOutputDataType();
    // dtype of indices output should be int32
    output_blobs_[1]->GetBlobDesc().data_type = DATA_TYPE_INT32;
    return status;
}

Status TopKLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);
    
    auto layer_param = dynamic_cast<TopKLayerParam*>(param_);
    CHECK_PARAM_NULL(layer_param);

    Blob* input_blob = input_blobs_[0];
    auto input_dims = input_blob->GetBlobDesc().dims;
    auto output_dims = input_dims;

    int axis = layer_param->axis;
    if (axis < 0) {
        axis += (int)input_blob->GetBlobDesc().dims.size();
        layer_param->axis = axis;
    }
    if (axis < 0 || axis > input_blob->GetBlobDesc().dims.size()) {
        LOGE_IF(!ignore_error, "Error: TopKLayer axis(%d) is invalid\n", axis);
        return Status(TNNERR_PARAM_ERR, "TopKLayer axis is invalid");
    }

    if (output_blobs_.size() != 2) {
        return Status(TNNERR_PARAM_ERR, "TopKLayer output blobs size != 2");
    }

    if (input_blobs_.size() > 1) {
        return Status(TNNERR_PARAM_ERR, "TopKLayer only support one input blob, not support k as a input blob");
    }

    if (layer_param->k > 0) {
        output_dims[layer_param->axis] = std::min(layer_param->k, input_dims[layer_param->axis]);
    }
    output_blobs_[0]->GetBlobDesc().dims = output_dims;
    output_blobs_[1]->GetBlobDesc().dims = output_dims;
    return TNN_OK;
}

REGISTER_LAYER(TopK, LAYER_TOPK);

}  // namespace TNN_NS
