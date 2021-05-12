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

#include "tnn/layer/elementwise_layer.h"

namespace TNN_NS {

DECLARE_LAYER(LayerNorm, LAYER_LAYER_NORM);

Status LayerNormLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status LayerNormLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);
    
    //check dims
    auto layer_param = dynamic_cast<LayerNormLayerParam *>(param_);
    RETURN_VALUE_ON_NEQ(!layer_param, false, Status(TNNERR_PARAM_ERR, "LayerNormLayerParam is nil"));
    if (input_blobs_.size() < 3) {
        return Status(TNNERR_PARAM_ERR, "LayerNormLayer has no input blob of scale or bias");
    }
    
    auto input_blob  = input_blobs_[0];
    auto scale_blob  = input_blobs_[1];
    auto bias_blob  = input_blobs_[2];
    auto dims_input = input_blob->GetBlobDesc().dims;
    auto dims_scale = scale_blob->GetBlobDesc().dims;
    auto dims_bias = bias_blob->GetBlobDesc().dims;

    if (layer_param->reduce_dims_size != dims_scale.size() || !DimsVectorUtils::Equal(dims_scale, dims_bias)) {
        return Status(TNNERR_PARAM_ERR, "LayerNormLayer has invalid dims for input blob of scale or bias");
    }
    
    //enure dims are valid
    const int dim_offset = (int)dims_input.size() - (int)dims_scale.size();
    for (int i=0; i<dims_scale.size(); i++) {
        if (dim_offset < 0 || dims_input[i + dim_offset] != dims_scale[i] || dims_scale[i] != dims_bias[i]) {
            return Status(TNNERR_PARAM_ERR, "LayerNormLayer has invalid dims for input blob");
        }
    }
    
    output_blobs_[0]->GetBlobDesc().dims =dims_input;
    
    return TNN_OK;
}

REGISTER_LAYER(LayerNorm, LAYER_LAYER_NORM);

}  // namespace TNN_NS
