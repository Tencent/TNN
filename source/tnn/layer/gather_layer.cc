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
DECLARE_LAYER(Gather, LAYER_GATHER);

Status GatherLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status GatherLayer::InferOutputShape() {
    const auto input_blob = input_blobs_[0];
    const auto input_dims = input_blob->GetBlobDesc().dims;
    auto output_blob      = output_blobs_[0];
    output_blob->GetBlobDesc().dims = input_dims;
    // the output blob has only one dim, the value is the size of input blob dims
    auto layer_param    = dynamic_cast<GatherLayerParam*>(param_);
    auto layer_resource = dynamic_cast<GatherLayerResource*>(resource_);
    auto indices_dims  = layer_resource->indices.GetBufferDims();
    auto axis           = layer_param->axis;
    if (indices_dims.size() > 1) {
        LOGE("Gather: no support indices dims > 1\n");
        return TNNERR_UNSUPPORT_NET;
    }
    int indices_count = DimsVectorUtils::Count(indices_dims);
    output_blob->GetBlobDesc().dims[axis] = indices_count;
    return TNN_OK;
}

REGISTER_LAYER(Gather, LAYER_GATHER);

}  // namespace TNN_NS