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

DECLARE_LAYER(StrideSlice, LAYER_STRIDED_SLICE);

Status StrideSliceLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status StrideSliceLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);
    
    StrideSliceLayerParam* layer_param = dynamic_cast<StrideSliceLayerParam*>(param_);
    if (!layer_param) {
        LOGE_IF(!ignore_error, "StrideSliceLayer param is nil\n");
        return Status(TNNERR_PARAM_ERR, "StrideSliceLayer param is nil");
    }

    Blob* input_blob  = input_blobs_[0];
    Blob* output_blob = output_blobs_[0];

    output_blob->GetBlobDesc().dims.clear();
    auto input_dims = input_blob->GetBlobDesc().dims;

    if (layer_param->begins.size() != input_dims.size() || layer_param->ends.size() != input_dims.size() ||
        layer_param->strides.size() != input_dims.size()) {
        LOGE_IF(!ignore_error, "StrideSliceLayer param got wrong size: input dims size: %ld\n", input_dims.size());
        return Status(TNNERR_PARAM_ERR, "StrideSliceLayer param got wrong size");
    }

    auto begins = layer_param->begins;
    std::reverse(begins.begin(), begins.end());

    auto ends = layer_param->ends;
    std::reverse(ends.begin(), ends.end());

    auto strides = layer_param->strides;
    std::reverse(strides.begin(), strides.end());

    auto sizes = strides;

    if (input_blobs_.size() > 1) {
        // NCNN crop layer, reference mode :
        // blob[0].size = blob[1].size
        //前闭后开区间
        sizes = input_blobs_[1]->GetBlobDesc().dims;
        for (int i = 0; i < input_dims.size(); i++) {
            ends[i] = sizes[i] + begins[i];
            if (ends[i] > input_dims[i]) {
                LOGE_IF(!ignore_error, "StrideSliceLayer param is invalid. Check NCNN Param\n");
                return Status(TNNERR_PARAM_ERR, "StrideSliceLayer param is invalid. Check NCNN Param");
            }
        }

    } else {
        //前闭后开区间
        for (int i = 0; i < input_dims.size(); i++) {
            if (begins[i] < 0) {
                begins[i] += input_blob->GetBlobDesc().dims[i];
            }
            if (ends[i] == 0) {
                ends[i] = input_dims[i];
            }

            if (ends[i] < 0) {
                ends[i] += input_dims[i];
            }

            if (begins[i] >= ends[i]) {
                LOGE_IF(!ignore_error, "StrideSliceLayer param is invalid\n");
                return Status(TNNERR_PARAM_ERR, "StrideSliceLayer param is invalid");
            }

            sizes[i] = (ends[i] - begins[i] - 1) / strides[i] + 1;

            if (sizes[i] <= 0) {
                LOGE_IF(!ignore_error, "StrideSliceLayer param is invalid\n");
                return Status(TNNERR_PARAM_ERR, "StrideSliceLayer param is invalid");
            }
        }
    }

    output_blob->GetBlobDesc().dims = sizes;

    return TNN_OK;
}

REGISTER_LAYER(StrideSlice, LAYER_STRIDED_SLICE);

}  // namespace TNN_NS
