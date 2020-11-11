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

namespace TNN_NS {

DECLARE_LAYER(StrideSliceV2, LAYER_STRIDED_SLICE_V2);

Status StrideSliceV2Layer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status StrideSliceV2Layer::InferOutputShape() {
    BaseLayer::InferOutputShape();
    
    StrideSliceV2LayerParam* layer_param = dynamic_cast<StrideSliceV2LayerParam*>(param_);
    if (!layer_param) {
        LOGE("StrideSliceV2Layer param is nil\n");
        return Status(TNNERR_PARAM_ERR, "StrideSliceV2Layer param is nil");
    }

    Blob* input_blob  = input_blobs_[0];
    Blob* output_blob = output_blobs_[0];

    output_blob->GetBlobDesc().dims.clear();
    auto input_dims = input_blob->GetBlobDesc().dims;

    auto begins = layer_param->begins;
    auto ends = layer_param->ends;
    auto axes = layer_param->axes;
    auto strides = layer_param->strides;

    auto sizes = input_dims;

    //前闭后开区间
    for (int i = 0; i < axes.size(); i++) {
        int index = axes[i];
        if (begins[i] < 0) {
            begins[i] += input_blob->GetBlobDesc().dims[index];
        }

        if (ends[i] == INT_MAX) {
            ends[i] = input_dims[index];
        }

        if (ends[i] < 0) {
            ends[i] += input_dims[index];
        }

        if (begins[i] >= ends[i]) {
            LOGE("StrideSliceV2Layer param is invalid\n");
            return Status(TNNERR_PARAM_ERR, "StrideSliceV2Layer param is invalid");
        }

        sizes[index] = (ends[i] - begins[i] - 1) / strides[i] + 1;

        if (sizes[index] <= 0) {
            LOGE("StrideSliceV2Layer param is invalid\n");
            return Status(TNNERR_PARAM_ERR, "StrideSliceV2Layer param is invalid");
        }
    }

    layer_param->begins = begins;
    layer_param->ends = ends;
    output_blob->GetBlobDesc().dims = sizes;

    return TNN_OK;
}

REGISTER_LAYER(StrideSliceV2, LAYER_STRIDED_SLICE_V2);

}  // namespace TNN_NS
