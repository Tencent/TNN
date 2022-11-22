// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "tnn/layer/base_layer.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_LAYER(FlattenTorch, LAYER_FLATTENTORCH);

Status FlattenTorchLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status FlattenTorchLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);

    FlattenTorchLayerParam* param = dynamic_cast<FlattenTorchLayerParam*>(param_);
    CHECK_PARAM_NULL(param);

    Blob* input_blob  = input_blobs_[0];
    Blob* output_blob = output_blobs_[0];

    const DimsVector input_dims = input_blob->GetBlobDesc().dims;
    const int input_dims_size = input_dims.size();
    int start_dim = param->start_dim < 0 ? param->start_dim + input_dims_size : param->start_dim;
    int end_dim   = param->end_dim < 0 ? param->end_dim + input_dims_size : param->end_dim;
    if (start_dim > input_dims_size || end_dim > input_dims_size) {
        LOGE_IF(!ignore_error, "FlattenTorch Param Error! start_dim, end_dims > input num dims.\n");
        return Status(TNNERR_PARAM_ERR, "FlattenTorch param error, start_dim or end_dim > number_of input dims.");
    }
    if (start_dim > end_dim) {
        LOGE_IF(!ignore_error, "FlattenTorch Param Error! start_dim > end_dims.\n");
        return Status(TNNERR_PARAM_ERR, "FlattenTorch param error, start_dim > end_dim.");
    }

    DimsVector output_dims;
    for (int i=0; i<start_dim; i++) {
        output_dims.push_back(input_dims[i]);
    }
    if (start_dim < end_dim) {
        int flattened_dim = 1;
        for (int i=start_dim; i<end_dim+1; i++) {
            flattened_dim *= input_dims[i];
        }
        output_dims.push_back(flattened_dim);
    }
    for (int i=end_dim+1; i<input_dims_size; i++) {
        output_dims.push_back(input_dims[i]);
    }
    ///////////////////////////// 
    /*
    std::cout << "[FlattenTorch InferShape] in.name = " << input_blob->GetBlobDesc().name << ", out.name = " << output_blob->GetBlobDesc().name << std::endl;
    std::cout << "[FlattenTorch InferShape] start_dim = " << start_dim << ", end_dim = " << end_dim << " ===" << std::endl;
    std::cout << "[FlattenTorch InferShape], input_dims = [";
    for (int i=0; i<input_dims.size(); i++)
        std::cout << input_dims[i] << ",";
    std::cout << "] ===" << std::endl;
    
    std::cout << "[FlattenTorch InferShape], output_dims = [";
    for (int i=0; i<output_dims.size(); i++)
        std::cout << output_dims[i] << ",";
    std::cout << "] ===" << std::endl;
    */
    ///////////////////////////// 

    output_blob->GetBlobDesc().dims = output_dims;

    return TNN_OK;
}

REGISTER_LAYER(FlattenTorch, LAYER_FLATTENTORCH);

}  // namespace TNN_NS
