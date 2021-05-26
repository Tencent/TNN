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
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_CUDA_ACC(Squeeze, LAYER_SQUEEZE);

Status CudaSqueezeLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CudaSqueezeLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaSqueezeLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];
    auto dims = input_blob->GetBlobDesc().dims;
    int count = DimsVectorUtils::Count(dims);
    void* input_data = input_blob->GetHandle().base;
    void* output_data = output_blob->GetHandle().base;
    auto size = count * DataTypeUtils::GetBytesSize(input_blob->GetBlobDesc().data_type);
    CUDA_CHECK(cudaMemcpyAsync(output_data, input_data, size, cudaMemcpyDeviceToDevice, context_->GetStream()));
    return TNN_OK;
}

REGISTER_CUDA_ACC(Squeeze, LAYER_SQUEEZE);

}  // namespace TNN_NS
