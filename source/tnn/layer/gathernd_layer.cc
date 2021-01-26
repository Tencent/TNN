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
DECLARE_LAYER(GatherND, LAYER_GATHERND);

Status GatherNDLayer::InferOutputDataType() {
    BaseLayer::InferOutputDataType();
    
//    auto layer_param = dynamic_cast<GatherLayerParam*>(param_);
//    CHECK_PARAM_NULL(layer_param);
//    auto layer_resource = dynamic_cast<GatherLayerResource*>(resource_);
//    if ((layer_param->data_in_resource || layer_param->indices_in_resource) && !layer_resource) {
//        return Status(TNNERR_MODEL_ERR, "Gather resource is invalid");
//    }
//
//    //修改indices输入 data type
//    if (!layer_param->indices_in_resource) {
//        (*(input_blobs_.rbegin()))->GetBlobDesc().data_type = DATA_TYPE_INT32;
//    }
//
//    //修改输出data type
//    if (layer_param->data_in_resource) {
//        output_blobs_[0]->GetBlobDesc().data_type = layer_resource->data.GetDataType();
//    }
    
    return TNN_OK;
}

Status GatherNDLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);
    
    auto layer_param = dynamic_cast<GatherNDLayerParam*>(param_);
    CHECK_PARAM_NULL(layer_param);
    
    auto data_dims = input_blobs_[0]->GetBlobDesc().dims;
    auto indices_dims = input_blobs_[1]->GetBlobDesc().dims;
    
    int dim_index = 0;
    DimsVector output_dims;
    while (dim_index < indices_dims.size()-1) {
        output_dims.push_back(indices_dims[dim_index]);
        dim_index++;
    }
    
    dim_index  = indices_dims[indices_dims.size() -1];
    while (dim_index < data_dims.size()) {
        output_dims.push_back(data_dims[dim_index]);
        dim_index++;
    }
    
    output_blobs_[0]->GetBlobDesc().dims = output_dims;
    return TNN_OK;
}

REGISTER_LAYER(GatherND, LAYER_GATHERND);

}  // namespace TNN_NS
