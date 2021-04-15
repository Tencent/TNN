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
DECLARE_LAYER(Unsqueeze, LAYER_UNSQUEEZE);

Status UnsqueezeLayer::InferOutputDataType() {
    auto status = BaseLayer::InferOutputDataType();
    if (runtime_model_ != RUNTIME_MODE_CONST_FOLD) {
        return status;
    }
    const auto& input_name = input_blobs_[0]->GetBlobDesc().name;
    const auto& const_res  = const_resource_;
    if (const_res != nullptr && const_res->find(input_name) != const_res->end()) {
        output_blobs_[0]->SetFlag(output_blobs_[0]->GetFlag() | DATA_FLAG_ALLOCATE_IN_FORWARD);
    }
    return status;
}

Status UnsqueezeLayer::InferOutputShape(bool ignore_error) {
    auto status = BaseLayer::InferOutputShape(ignore_error);
    RETURN_ON_NEQ(status, TNN_OK);
    
    auto *layer_param = dynamic_cast<UnsqueezeLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    
    const auto input_dims  = input_blobs_[0]->GetBlobDesc().dims;
    // the output blob has only one dim, the value is the size of input blob dims
    
    auto axes = layer_param->axes;
    auto output_dims = input_dims;
    for (auto iter = axes.begin(); iter != axes.end(); iter++) {
        //Note: here it is diffreent from SqueezeLayer
        int axis = *iter;
        axis = axis < 0 ? axis + (int)output_dims.size() + 1 : axis;
        if (axis < 0 || axis > output_dims.size()) {
            return Status(TNNERR_PARAM_ERR, "UnsqueezeLayer has invalid input axes");
        }
        output_dims.insert(output_dims.begin() + axis, 1);
    }
    
    output_blobs_[0]->GetBlobDesc().dims = output_dims;
    return TNN_OK;
}

REGISTER_LAYER(Unsqueeze, LAYER_UNSQUEEZE);

}  // namespace TNN_NS
