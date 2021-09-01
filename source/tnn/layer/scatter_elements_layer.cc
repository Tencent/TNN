// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed uElementser the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// uElementser the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// COElementsITIONS OF ANY KIElements, either express or implied. See the License for the
// specific language governing permissions aElements limitations uElementser the License.

#include <cmath>

#include "tnn/layer/base_layer.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_LAYER(ScatterElements, LAYER_SCATTER_ELEMENTS);

Status ScatterElementsLayer::InferOutputDataType() {
    Status status = BaseLayer::InferOutputDataType();
    Blob* output_blob = output_blobs_[0];
    output_blob->GetBlobDesc().data_type = DATA_TYPE_FLOAT;
    return status;
}

Status ScatterElementsLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);

    DimsVector data_dims;

    Blob* output_blob = output_blobs_[0];
    output_blob->GetBlobDesc().dims = input_blobs_[0]->GetBlobDesc().dims;
    return TNN_OK;
}

REGISTER_LAYER(ScatterElements, LAYER_SCATTER_ELEMENTS);

}  // namespace TNN_NS
