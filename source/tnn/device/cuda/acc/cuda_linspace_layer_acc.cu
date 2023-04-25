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

#include "tnn/device/cuda/acc/cuda_layer_acc.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_CUDA_ACC(Linspace, LAYER_LINSPACE);

Status CudaLinspaceLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status status = CudaLayerAcc::Init(context, param, resource, inputs, outputs);
    RETURN_ON_FAIL(status);

    float eps = 1e-6;
    // Cuda Layer only support start and end is int
    auto layer_param = dynamic_cast<LinspaceLayerParam*>(param_);
    CHECK_PARAM_NULL(layer_param);

    float start = layer_param->start.f;
    float end = layer_param->end.f;
    if (layer_param->start_index != -1 && fabs(start - static_cast<int>(start)) > eps) {
        LOGE("Cuda Linspace Layer got non-int start\n");
        return TNNERR_LAYER_ERR;
    }

    if (layer_param->end_index != -1 && fabs(end - static_cast<int>(end)) > eps) {
        LOGE("Cuda Linspace Layer got non-int end\n");
        return TNNERR_LAYER_ERR;
    }

    return status;
}

Status CudaLinspaceLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaLinspaceLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

REGISTER_CUDA_ACC(Linspace, LAYER_LINSPACE);

}  // namespace TNN_NS
