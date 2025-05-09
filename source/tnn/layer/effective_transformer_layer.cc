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

DECLARE_LAYER(EffectiveTransformer, LAYER_EFFECTIVE_TRANSFORMER);

Status EffectiveTransformerLayer::InferOutputDataType() {
    BaseLayer::InferOutputDataType();
    for (int i = 0; i < output_blobs_.size(); ++i) {
        if (i == 0) {
            output_blobs_[i]->GetBlobDesc().data_type = input_blobs_[i]->GetBlobDesc().data_type;
        } else {
            // for offset
            output_blobs_[i]->GetBlobDesc().data_type = DATA_TYPE_INT32;
        }
    }
    return TNN_OK;
}

Status EffectiveTransformerLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);
    int rank = input_blobs_[0]->GetBlobDesc().dims.size();
    auto in_dims = input_blobs_[0]->GetBlobDesc().dims;
    output_blobs_[0]->GetBlobDesc().dims = in_dims;

    auto eff_param = dynamic_cast<EffectiveTransformerLayerParam*>(param_);
    CHECK_PARAM_NULL(eff_param);

    if (eff_param->is_remove_padding) {
        if (output_blobs_.size() != 3) {
            LOGE("Error: EffectiveTransformerLayer output number error.\n");
            return Status(TNNERR_PARAM_ERR, "Error: EffectiveTransformerLayer output number error.");
        }
        if (rank < 2) {
            LOGE("Error: EffectiveTransformerLayer input dims error.\n");
            return Status(TNNERR_PARAM_ERR, "Error: EffectiveTransformerLayer input dims error.");
        } else {
            int dim = 1;
            for (int i = 0; i < rank - 1; ++i) {
                dim *= in_dims[i];
            }
            dim += 1;   // token_number
            output_blobs_[1]->GetBlobDesc().dims = {dim};
            output_blobs_[2]->GetBlobDesc().dims = {in_dims[0] + 1};
        }
    } else {
        if (output_blobs_.size() == 2) {
            // for control flow
            output_blobs_[1]->GetBlobDesc().dims = {1};
        }
    }
    return TNN_OK;
}

REGISTER_LAYER(EffectiveTransformer, LAYER_EFFECTIVE_TRANSFORMER);

}  // namespace TNN_NS
