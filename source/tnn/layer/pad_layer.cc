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

#include <cmath>

#include "tnn/layer/base_layer.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {
DECLARE_LAYER(Pad, LAYER_PAD);

Status PadLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status PadLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);
    
    auto layer_param = dynamic_cast<PadLayerParam*>(param_);
    if (!layer_param) {
        LOGE_IF(!ignore_error, "Error: layer param is nil\n");
        return Status(TNNERR_PARAM_ERR, "Error: layer param is nil");
    }

    Blob* input_blob  = input_blobs_[0];
    Blob* output_blob = output_blobs_[0];
    auto dims         = input_blob->GetBlobDesc().dims;
    dims[3] += layer_param->pads[0] + layer_param->pads[1];
    dims[2] += layer_param->pads[2] + layer_param->pads[3];
    dims[1] += layer_param->pads[4] + layer_param->pads[5];

    output_blob->GetBlobDesc().dims = dims;
    return TNN_OK;
}

REGISTER_LAYER(Pad, LAYER_PAD);

}  // namespace TNN_NS
