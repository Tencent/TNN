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

#include "tnn/layer/base_layer.h"

namespace TNN_NS {

DECLARE_LAYER(DetectionOutput, LAYER_DETECTION_OUTPUT);

Status DetectionOutputLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status DetectionOutputLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);
    
    Blob* input_blob  = input_blobs_[0];
    Blob* output_blob = output_blobs_[0];

    DetectionOutputLayerParam* param = dynamic_cast<DetectionOutputLayerParam*>(param_);
    CHECK_PARAM_NULL(param);

    ASSERT(input_blobs_.size() > 2);
    //
    ASSERT(input_blobs_[0]->GetBlobDesc().dims[0] == input_blobs_[1]->GetBlobDesc().dims[0]);
    // get input node 2 height
    int num_priors      = input_blobs_[2]->GetBlobDesc().dims[2] / 4;
    int num_loc_classes = param->share_location ? 1 : param->num_classes;

    ASSERT(num_priors * num_loc_classes * 4 == input_blobs_[0]->GetBlobDesc().dims[1]);
    ASSERT(num_priors * param->num_classes == input_blobs_[1]->GetBlobDesc().dims[1]);

    // num() and channels() are 1.
    std::vector<int> output_dims(2, 1);
    // Since the number of bboxes to be kept is unknown before nms, we manually
    // set it to (fake) 1.
    output_dims.push_back(param->keep_top_k);
    // Each row is a 7 dimension vector, which stores
    // [image_id, label, confidence, xmin, ymin, xmax, ymax]
    output_dims.push_back(7);
    output_blob->GetBlobDesc().dims = output_dims;
    return TNN_OK;
}

REGISTER_LAYER(DetectionOutput, LAYER_DETECTION_OUTPUT);

}  // namespace TNN_NS
