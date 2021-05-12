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
#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {

DECLARE_CUDA_ACC(Histogram, LAYER_HISTOGRAM);

__global__ void histogram_kernel(const int* input, int* output, int count) {
    CUDA_KERNEL_LOOP(i, count) {
        int index = input[i];
        atomicAdd(&output[index], 1);
    }
}

Status CudaHistogramLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);;
}

Status CudaHistogramLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaHistogramLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input_data_type  = inputs[0]->GetBlobDesc().data_type;
    auto input_data = (int *)(inputs[0]->GetHandle().base);
    auto output_data = (int *)(outputs[0]->GetHandle().base);

    const int ele_size = DataTypeUtils::GetBytesSize(outputs[0]->GetBlobDesc().data_type);
    const int count = DimsVectorUtils::Count(inputs[0]->GetBlobDesc().dims);
    
    cudaMemset(output_data, 0, ele_size * DimsVectorUtils::Count(outputs[0]->GetBlobDesc().dims));
    histogram_kernel<<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
        input_data, output_data, count);

    return TNN_OK;
}

REGISTER_CUDA_ACC(Histogram, LAYER_HISTOGRAM);

}  // namespace TNN_NS
