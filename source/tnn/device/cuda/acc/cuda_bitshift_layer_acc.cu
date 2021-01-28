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
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

DECLARE_CUDA_ACC(BitShift, LAYER_BITSHIFT);

__global__ void bitshift_kernel(const int* input, int* output, int count, int bits, int direction) {
    CUDA_KERNEL_LOOP(index, count) {
        if (direction == 0) {
            output[index] = input[index] >> bits;
        } else {
            output[index] = input[index] << bits;
        }
    }
}

Status CudaBitShiftLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);;
}

Status CudaBitShiftLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaBitShiftLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<BitShiftLayerParam*>(param_);
    CHECK_PARAM_NULL(layer_param);
    
    auto input_data_type  = inputs[0]->GetBlobDesc().data_type;
    auto input_data = (int*)(inputs[0]->GetHandle().base);
    auto output_data = (int *)(outputs[0]->GetHandle().base);

    const int count = DimsVectorUtils::Count(inputs[0]->GetBlobDesc().dims);
    bitshift_kernel<<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
            input_data, output_data, count, layer_param->bits, layer_param->direction);

    return TNN_OK;
}

REGISTER_CUDA_ACC(BitShift, LAYER_BITSHIFT);

}  // namespace TNN_NS
