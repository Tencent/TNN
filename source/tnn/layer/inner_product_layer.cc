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
DECLARE_LAYER(InnerProduct, LAYER_INNER_PRODUCT);

Status InnerProductLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status InnerProductLayer::InferOutputShape() {
    InnerProductLayerParam* ip_param = dynamic_cast<InnerProductLayerParam*>(param_);
    CHECK_PARAM_NULL(ip_param);

    Blob* input_blob  = input_blobs_[0];
    Blob* output_blob = output_blobs_[0];

    int N    = ip_param->num_output;
    int axis = ip_param->axis;
    //    int M    = DimsVectorUtils::Count(input_blob->GetBlobDesc().dims, 0, axis);
    //    int K    = DimsVectorUtils::Count(input_blob->GetBlobDesc().dims, axis);

    output_blob->GetBlobDesc().dims = input_blob->GetBlobDesc().dims;

    output_blob->GetBlobDesc().dims[axis] = N;
    for (int i = axis + 1; i < output_blob->GetBlobDesc().dims.size(); i++) {
        output_blob->GetBlobDesc().dims[i] = 1;
    }

    return TNN_OK;
}

REGISTER_LAYER(InnerProduct, LAYER_INNER_PRODUCT);

}  // namespace TNN_NS
