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
DECLARE_LAYER(Squeeze, LAYER_SQUEEZE);

Status SqueezeLayer::InferOutputDataType() {
    auto status = BaseLayer::InferOutputDataType();
    if (runtime_model_ != RUNTIME_MODE_CONST_FOLD) {
        return status;
    }
    const auto& input_name = input_blobs_[0]->GetBlobDesc().name;
    const auto& const_res  = const_resource_;
    if (const_res != nullptr && const_res->find(input_name) != const_res->end()) {
        output_blobs_[0]->flag = output_blobs_[0]->flag | DATA_FLAG_ALLOCATE_IN_FORWARD;
    }
    return status;
}

Status SqueezeLayer::InferOutputShape(bool ignore_error) {
    auto status = BaseLayer::InferOutputShape(ignore_error);
    auto layer_param        = dynamic_cast<SqueezeLayerParam*>(param_);
    const auto& output_blob = output_blobs_[0];
    DimsVector input_dims   = input_blobs_[0]->GetBlobDesc().dims;
    DimsVector output_dims  = input_dims;
    RETURN_VALUE_ON_NEQ(input_dims.size() > 0, true, Status(TNNERR_PARAM_ERR, "SqueezeLayer has invalid input size"));
    auto axes = layer_param->axes;
    for (auto axis : axes) {
        axis = axis < 0 ? axis + output_dims.size() : axis;
        if (axis < 0 || axis >= output_dims.size() || output_dims[axis] != 1) {
            return Status(TNNERR_PARAM_ERR, "SqueezeLayer has invalid input axes");
        }
        output_dims.erase(output_dims.begin() + axis);
    }
    output_blob->GetBlobDesc().dims = output_dims;
    return status;
}

REGISTER_LAYER(Squeeze, LAYER_SQUEEZE);

}  // namespace TNN_NS
