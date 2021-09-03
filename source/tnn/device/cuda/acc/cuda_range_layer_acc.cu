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

DECLARE_CUDA_ACC(Range, LAYER_RANGE);

Status CudaRangeLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CudaRangeLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaRangeLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

REGISTER_CUDA_ACC(Range, LAYER_RANGE);

}  // namespace TNN_NS
