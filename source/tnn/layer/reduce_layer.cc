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

#include "reduce_layer.h"

#include <set>

#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

Status ReduceLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);

    auto layer_param = dynamic_cast<ReduceLayerParam*>(param_);
    if (!layer_param) {
        LOGE_IF(!ignore_error, "Error: Reduce may not support axes != 1, depend on device\n");
        return Status(TNNERR_MODEL_ERR, "Error: Reduce may not support axes != 1, depend on device");
    }

    Blob* input_blob  = input_blobs_[0];
    Blob* output_blob = output_blobs_[0];
    auto dims         = input_blob->GetBlobDesc().dims;

    if (layer_param->axis.size() == 0) {
        layer_param->all_reduce = 1;
    }

    if (layer_param->all_reduce) {
        layer_param->axis.clear();
        for (int i = 0; i < dims.size(); ++i) {
            layer_param->axis.push_back(i);
        }
    }

    std::set<int> axis_filter;
    for (auto& axis : layer_param->axis) {
        axis = axis >= 0 ? axis : axis + (int)dims.size();
        if (axis < 0 || axis >= dims.size()) {
            LOGE_IF(!ignore_error, "Error: layer param axis is invalid\n");
            return Status(TNNERR_MODEL_ERR, "Error: layer param axis is invalid");
        }
        dims[axis] = 1;
        axis_filter.insert(axis);
    }

    DimsVector output_dims;
    if (layer_param->keep_dims == 0) {
        for (int i = 0; i < dims.size(); ++i) {
            if (axis_filter.count(i) == 0) {
                output_dims.push_back(dims[i]);
            }
        }
    } else {
        output_dims = dims;
    }

    output_blob->GetBlobDesc().dims = output_dims;

    return TNN_OK;
}
}  // namespace TNN_NS
