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
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {
DECLARE_LAYER(Where, LAYER_WHERE);

Status WhereLayer::InferOutputDataType() {
    auto status = BaseLayer::InferOutputDataType();
    RETURN_ON_NEQ(status, TNN_OK);
    
    output_blobs_[0]->GetBlobDesc().data_type = input_blobs_[0]->GetBlobDesc().data_type;
    return TNN_OK;
}

Status WhereLayer::InferOutputShape(bool ignore_error) {
    //X, Y, condition order for input
    auto status = BaseLayer::InferOutputShape(ignore_error);
    RETURN_ON_NEQ(status, TNN_OK);
    
    Blob* input_blob = input_blobs_[0];
    auto dims = input_blob->GetBlobDesc().dims;
    auto dims_output = dims;
    for (auto iter : input_blobs_) {
        dims       = iter->GetBlobDesc().dims;
        dims_output = DimsVectorUtils::Max(dims, dims_output);
    }

    output_blobs_[0]->GetBlobDesc().dims = dims_output;
    return TNN_OK;
}
REGISTER_LAYER(Where, LAYER_WHERE);

}  // namespace TNN_NS
