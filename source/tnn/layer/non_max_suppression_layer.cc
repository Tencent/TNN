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

#include <algorithm>

#include "tnn/layer/base_layer.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_LAYER(NonMaxSuppression, LAYER_NON_MAX_SUPPRESSION);

Status NonMaxSuppressionLayer::InferOutputDataType() {
    Status status = BaseLayer::InferOutputDataType();
    RETURN_ON_NEQ(status, TNN_OK);

    output_blobs_[0]->GetBlobDesc().data_type = DATA_TYPE_INT32;
    return TNN_OK;
}

Status NonMaxSuppressionLayer::InferOutputShape(bool ignore_error) {
    auto status = BaseLayer::InferOutputShape(ignore_error);
    RETURN_ON_NEQ(status, TNN_OK);

    auto layer_param = dynamic_cast<NonMaxSuppressionLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);

    Blob *boxes_blob  = input_blobs_[0];
    Blob *scores_blob = input_blobs_[1];
    Blob *output_blob = output_blobs_[0];

    auto boxes_dims  = boxes_blob->GetBlobDesc().dims;
    auto scores_dims = scores_blob->GetBlobDesc().dims;

    int64_t output_dim_max_box = layer_param->max_output_boxes_per_class;
    if (output_dim_max_box > boxes_dims[1]) {
        output_dim_max_box = boxes_dims[1];
    }

    int last_dim     = 3;
    auto output_dims = {(int)output_dim_max_box, last_dim};

    output_blob->GetBlobDesc().dims = output_dims;

    return TNN_OK;
}

REGISTER_LAYER(NonMaxSuppression, LAYER_NON_MAX_SUPPRESSION);

}  // namespace TNN_NS
