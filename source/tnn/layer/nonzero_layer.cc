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

#include "base_layer.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {
DECLARE_LAYER(NonZero, LAYER_NONZERO);

Status NonZeroLayer::InferOutputDataType() {
    BaseLayer::InferOutputDataType();
    
    for (auto& iter : output_blobs_) {
        int allocate_status = DATA_FLAG_ALLOCATE_IN_FORWARD;
        if (runtime_model_ == RUNTIME_MODE_NORMAL &&
            const_resource_ != nullptr && const_resource_->find(iter->GetBlobDesc().name) != const_resource_->end()) {
            allocate_status = 0;
        }
        iter->flag = iter->flag | allocate_status;
        iter->GetBlobDesc().data_type = DATA_TYPE_INT32;
    }
    return TNN_OK;
}

Status NonZeroLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);
    
    auto input_dims  = input_blobs_[0]->GetBlobDesc().dims;
    int input_dim_size = (int)input_dims.size();
    int count = DimsVectorUtils::Count(input_dims);
    
    output_blobs_[0]->GetBlobDesc().dims = {input_dim_size, count};
    return TNN_OK;
}

REGISTER_LAYER(NonZero, LAYER_NONZERO);

}  // namespace TNN_NS
