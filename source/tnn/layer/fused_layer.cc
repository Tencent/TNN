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

namespace TNN_NS {

DECLARE_LAYER(Fused, LAYER_FUSED);

Status FusedLayer::InferOutputDataType() {
    auto status = BaseLayer::InferOutputDataType();
    RETURN_ON_NEQ(status, TNN_OK);
 
    FusedLayerParam* layer_param = dynamic_cast<FusedLayerParam*>(param_);
    CHECK_PARAM_NULL(layer_param);

    if (layer_param->type == FusionType_None) {
        LOGE("Error: FusedLayer should have a definite fusion type in layer param.\n");
        return Status(TNNERR_PARAM_ERR, "Error: FusedLayer should have a definite fusion type in layer param.");
    } else if (layer_param->type == FusionType_TRTPlugin_BertQKVtoContextV1) {
        output_blobs_[0]->GetBlobDesc().data_type = input_blobs_[0]->GetBlobDesc().data_type;
    } else if (layer_param->type == FusionType_TRTPlugin_BertQKVtoContextV2) {
        output_blobs_[0]->GetBlobDesc().data_type = input_blobs_[0]->GetBlobDesc().data_type;
    } else if (layer_param->type == FusionType_TRTPlugin_BertQKVtoContextV3) {
        output_blobs_[0]->GetBlobDesc().data_type = input_blobs_[0]->GetBlobDesc().data_type;
    } else if (layer_param->type == FusionType_AddBiasResidualLayerNorm ||
               layer_param->type == FusionType_FFN ||
               layer_param->type == FusionType_Attention ||
               layer_param->type == FusionType_Flash_Attention ||
               layer_param->type == FusionType_Cross_Attention) {
        output_blobs_[0]->GetBlobDesc().data_type = input_blobs_[0]->GetBlobDesc().data_type;
    } else {
        LOGE("Error: FusedLayer fusion type not supported yet: %d.\n", (int)layer_param->type);
        return Status(TNNERR_PARAM_ERR, "Error: FusedLayer fusion type not supported yet.");
    }

    return TNN_OK;
}

Status FusedLayer::InferOutputShape(bool ignore_error) {
    //X, Y, condition order for input
    auto status = BaseLayer::InferOutputShape(ignore_error);
    RETURN_ON_NEQ(status, TNN_OK);

    FusedLayerParam* layer_param = dynamic_cast<FusedLayerParam*>(param_);
    CHECK_PARAM_NULL(layer_param);

    if (layer_param->type == FusionType_None) {
        LOGE_IF(!ignore_error, "Error: FusedLayer should have a definite fusion type in layer param.\n");
        return Status(TNNERR_PARAM_ERR, "Error: FusedLayer should have a definite fusion type in layer param.");
    } else if (layer_param->type == FusionType_TRTPlugin_BertQKVtoContextV1) {
        Blob* input_blob = input_blobs_[0];
        Blob* output_blob = output_blobs_[0];
 
        // Input Shape:  [Seq_len, Batch, 3*Hidden_Size, 1, 1]
        // Output Shape: [Seq_len, Batch, Hidden_Size, 1, 1]
        output_blob->GetBlobDesc().dims = input_blob->GetBlobDesc().dims;
        output_blob->GetBlobDesc().dims[2] /= 3;
 
        return TNN_OK;
    } else if (layer_param->type == FusionType_TRTPlugin_BertQKVtoContextV2) {
        Blob* input_blob = input_blobs_[0];
        Blob* output_blob = output_blobs_[0];
 
        // Input Shape:  [Seq_len*Batch compressed to dense mode, 3*Hidden_Size, 1, 1]
        // Output Shape: [Seq_len*Batch compressed to dense mode, Hidden_Size, 1, 1]
        output_blob->GetBlobDesc().dims = input_blob->GetBlobDesc().dims;
        output_blob->GetBlobDesc().dims[1] /= 3;
 
        return TNN_OK;
    } else if (layer_param->type == FusionType_TRTPlugin_BertQKVtoContextV3) {
        LOGE_IF(!ignore_error, "Error: FusedLayer Bert QKVtoContext V3 not supported yet.\n");
        return Status(TNNERR_PARAM_ERR, "Error: FusedLayer QKVtoContext V3 not supported yet.");
    } else if (layer_param->type == FusionType_AddBiasResidualLayerNorm ||
               layer_param->type == FusionType_FFN ||
               layer_param->type == FusionType_Attention || 
               layer_param->type == FusionType_Cross_Attention ||
               layer_param->type == FusionType_Flash_Attention) {
        Blob* input_blob = input_blobs_[0];
        Blob* output_blob = output_blobs_[0];

        output_blob->GetBlobDesc().dims = input_blob->GetBlobDesc().dims;
        return TNN_OK;
    } else {
        LOGE_IF(!ignore_error, "Error: FusedLayer fusion type not supported yet.\n");
        return Status(TNNERR_PARAM_ERR, "Error: FusedLayer fusion type not supported yet.");
    }
    
    return TNN_OK;
}

REGISTER_LAYER(Fused, LAYER_FUSED);

}  // namespace TNN_NS
