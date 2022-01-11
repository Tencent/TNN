// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_LAYER(MultiHeadAttention, LAYER_MULTI_HEAD_ATTENTION);

Status MultiHeadAttentionLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status MultiHeadAttentionLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);

    MultiHeadAttentionLayerParam* mha_param = dynamic_cast<MultiHeadAttentionLayerParam*>(param_);
    CHECK_PARAM_NULL(mha_param);

    Blob* input_q_blob = input_blobs_[0];
    Blob* input_k_blob = input_blobs_[1];
    Blob* input_v_blob = input_blobs_[2];
    Blob* output_blob = output_blobs_[0];

    const auto input_q_dims = input_q_blob->GetBlobDesc().dims;
    const auto input_k_dims = input_q_blob->GetBlobDesc().dims;
    const auto input_v_dims = input_q_blob->GetBlobDesc().dims;
    const int input_q_dims_size = input_q_dims.size();
    const int input_k_dims_size = input_k_dims.size();
    const int input_v_dims_size = input_v_dims.size();

    if (input_q_dims_size!=4 || 
        input_q_dims_size!=input_k_dims_size ||
        input_q_dims_size!=input_v_dims_size) {
        LOGE_IF(!ignore_error, "dim of multi head attention input qkv not equal to 4, [batch, seq, num_head, hidden_per_head]\n");
        return Status(TNNERR_PARAM_ERR, "multi head attention input size error");
    }
    for (int dim=0; dim<input_q_dims_size; dim++) {
        if (input_q_dims[dim]!=input_k_dims[dim] ||
            input_q_dims[dim]!=input_v_dims[dim]) {
            LOGE_IF(!ignore_error, "multi head attention input: dims of input q,k,v not equal\n");
            return Status(TNNERR_PARAM_ERR, "multi head attention input size error");
        }
    } 

    mha_param->num_heads = input_q_dims[2];
    mha_param->hidden_size = input_q_dims[2]*input_q_dims[3];
    output_blob->GetBlobDesc().dims = {input_q_dims[0], input_q_dims[1], mha_param->hidden_size};

    return TNN_OK;
}

REGISTER_LAYER(MultiHeadAttention, LAYER_MULTI_HEAD_ATTENTION);

}  // namespace TNN_NS
