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

namespace TNN_NS {

DECLARE_LAYER(GLU, LAYER_GLU);

Status GLULayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status GLULayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);

    Blob* input_blob  = input_blobs_[0];
    Blob* output_blob = output_blobs_[0];
    auto param        = dynamic_cast<GLULayerParam*>(param_);
    if (param->axis < 0) {
        param->axis += input_blob->GetBlobDesc().dims.size();
    }
    DimsVector input_dims  = input_blob->GetBlobDesc().dims;
    DimsVector output_dims = input_blob->GetBlobDesc().dims;
    const int axis         = param->axis;
    if (axis == 0 || input_dims[axis] % 2 != 0) {
        LOGE("GLULayer get wrong param\n");
        return {TNNERR_UNSUPPORT_NET, "GLULayer get wrong param"};
    }
    output_dims[param->axis]        = input_dims[param->axis] / 2;
    output_blob->GetBlobDesc().dims = output_dims;
    return TNN_OK;
}

REGISTER_LAYER(GLU, LAYER_GLU);

}  // namespace TNN_NS
