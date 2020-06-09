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
#include <cmath>

#include "tnn/layer/base_layer.h"

namespace TNN_NS {

DECLARE_LAYER(Permute, LAYER_PERMUTE);

Status PermuteLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status PermuteLayer::InferOutputShape() {
    PermuteLayerParam* permute_param = dynamic_cast<PermuteLayerParam*>(param_);
    CHECK_PARAM_NULL(permute_param);

    Blob* input_blob  = input_blobs_[0];
    Blob* output_blob = output_blobs_[0];

    output_blob->GetBlobDesc().dims.clear();
    auto input_dims          = input_blob->GetBlobDesc().dims;
    std::vector<int>& orders = permute_param->orders;

    for (int i = 0; i < input_dims.size(); ++i) {
        if (std::find(orders.begin(), orders.end(), i) == orders.end()) {
            orders.push_back(i);
        }
    }
    if (permute_param->orders.size() != input_dims.size()) {
        LOGE("Permute param got wrong size.\n");
        return Status(TNNERR_PARAM_ERR, "Permute param got wrong size");
    }

    for (int i = 0; i < permute_param->orders.size(); ++i) {
        int order = permute_param->orders[i];
        if (order < 0 || order > input_dims.size() - 1) {
            LOGE("Permute param out of range.\n");
            return Status(TNNERR_PARAM_ERR, "Permute param out of range");
        }
        output_blob->GetBlobDesc().dims.push_back(input_dims[order]);
    }

    return TNN_OK;
}

REGISTER_LAYER(Permute, LAYER_PERMUTE);

}  // namespace TNN_NS
