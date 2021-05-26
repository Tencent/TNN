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
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {
DECLARE_LAYER(Shape, LAYER_SHAPE);

Status ShapeLayer::InferOutputDataType() {
    ASSERT(output_blobs_.size() == 1);
    BaseLayer::InferOutputDataType();
    
    output_blobs_[0]->GetBlobDesc().data_type = DATA_TYPE_INT32;
    
    //insure no blob reuse in const forward, so set alloc status DATA_FLAG_ALLOCATE_IN_FORWARD
    for (auto& iter : output_blobs_) {
        iter->SetFlag(iter->GetFlag() | DATA_FLAG_CHANGE_IF_SHAPE_DIFFER | DATA_FLAG_ALLOCATE_IN_FORWARD);
    }
    return TNN_OK;
}

Status ShapeLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);
    
    ASSERT(input_blobs_.size() == 1);
    const auto& input_blob = input_blobs_[0];
    const auto& output_blob = output_blobs_[0];
    // the output blob has only one dim, the value is the size of input blob dims
    output_blob->GetBlobDesc().dims = {(int)input_blob->GetBlobDesc().dims.size()};
    return TNN_OK;
}

REGISTER_LAYER(Shape, LAYER_SHAPE);

}  // namespace TNN_NS
