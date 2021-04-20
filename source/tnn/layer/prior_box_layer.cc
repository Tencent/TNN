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

DECLARE_LAYER(PriorBox, LAYER_PRIOR_BOX);

Status PriorBoxLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status PriorBoxLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);
    
    Blob *input_blob                    = input_blobs_[0];
    Blob *output_blob                   = output_blobs_[0];
    PriorBoxLayerParam *prior_box_param = dynamic_cast<PriorBoxLayerParam *>(param_);
    CHECK_PARAM_NULL(prior_box_param);

    int num_priors = static_cast<int>(prior_box_param->aspect_ratios.size() * prior_box_param->min_sizes.size());
    if (!prior_box_param->max_sizes.empty()) {
        ASSERT(prior_box_param->min_sizes.size() == prior_box_param->max_sizes.size());
        for (int i = 0; i < prior_box_param->max_sizes.size(); ++i) {
            ASSERT(prior_box_param->max_sizes[i] > prior_box_param->min_sizes[i]);
            num_priors++;
        }
    }
    int num     = input_blob->GetBlobDesc().dims[0];
    int channel = input_blob->GetBlobDesc().dims[1];
    int height  = input_blob->GetBlobDesc().dims[2];
    int width   = input_blob->GetBlobDesc().dims[3];
    DimsVector output_dims;
    output_dims.push_back(1);
    output_dims.push_back(2);
    output_dims.push_back(height * width * num_priors * 4);
    // a little trick. hack the prior box output
    output_dims.push_back(1);
    output_blob->GetBlobDesc().dims = output_dims;
    return TNN_OK;
}

REGISTER_LAYER(PriorBox, LAYER_PRIOR_BOX);

}  // namespace TNN_NS
