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
DECLARE_LAYER(Gather, LAYER_GATHER);

Status GatherLayer::InferOutputDataType() {
    BaseLayer::InferOutputDataType();

    auto layer_param = dynamic_cast<GatherLayerParam*>(param_);
    CHECK_PARAM_NULL(layer_param);
    auto layer_resource = dynamic_cast<GatherLayerResource*>(resource_);
    if ((layer_param->data_in_resource || layer_param->indices_in_resource) && !layer_resource) {
        return Status(TNNERR_MODEL_ERR, "Gather resource is invalid");
    }

    //修改indices输入 data type
    if (!layer_param->indices_in_resource) {
        (*(input_blobs_.rbegin()))->GetBlobDesc().data_type = DATA_TYPE_INT32;
    }

    //修改输出data type
    if (layer_param->data_in_resource) {
        output_blobs_[0]->GetBlobDesc().data_type = layer_resource->data.GetDataType();
    }

    // if gather has 2 inputs, the output datatype is as same as the first input datatype
    if (input_blobs_.size() >= 2) {
        output_blobs_[0]->GetBlobDesc().data_type = input_blobs_[0]->GetBlobDesc().data_type;
    }

    return TNN_OK;
}

Status GatherLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);

    auto layer_param = dynamic_cast<GatherLayerParam*>(param_);
    CHECK_PARAM_NULL(layer_param);
    auto layer_resource = dynamic_cast<GatherLayerResource*>(resource_);
    if ((layer_param->data_in_resource || layer_param->indices_in_resource) && !layer_resource) {
        return Status(TNNERR_MODEL_ERR, "Gather resource is invalid");
    }

    DimsVector data_dims, indices_dims;
    if (layer_param->data_in_resource) {
        data_dims = layer_resource->data.GetBufferDims();
    } else {
        data_dims = (*(input_blobs_.begin()))->GetBlobDesc().dims;
    }

    if (layer_param->indices_in_resource) {
        indices_dims = layer_resource->indices.GetBufferDims();
    } else {
        indices_dims = (*(input_blobs_.rbegin()))->GetBlobDesc().dims;
    }

    int axis = layer_param->axis;
    while (axis < 0) {
        axis += data_dims.size();
    }
    layer_param->axis = axis;

    DimsVector output_dims;
    if (axis > 0 && axis < data_dims.size()) {
        output_dims.insert(output_dims.end(), data_dims.begin(), data_dims.begin() + axis);
    }

    output_dims.insert(output_dims.end(), indices_dims.begin(), indices_dims.end());

    if (axis < data_dims.size() - 1) {
        output_dims.insert(output_dims.end(), data_dims.begin() + axis + 1, data_dims.end());
    }

    output_blobs_[0]->GetBlobDesc().dims = output_dims;
    return TNN_OK;
}

REGISTER_LAYER(Gather, LAYER_GATHER);

}  // namespace TNN_NS
