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

DECLARE_CUDA_ACC(Pow, LAYER_POWER);

__global__ void pow_kernel(int n, const float* srcData, const float power,
        const float scale, const float shift, float* dstData) {
    CUDA_KERNEL_LOOP(index, n) {
        float result = 1;
        float input = srcData[index] * scale + shift;
        for (int i = 0; i < power; ++i) {
            result *= input;
        }
        dstData[index] = result;
    }
}

__global__ void pow_kernel_fp16(int n, bool odd, const __half2 * srcData, const float power,
        const float scale, const float shift, __half2 * dstData) {
#if __CUDA_ARCH__ > 520    
    CUDA_KERNEL_LOOP(index, n) {
        if (odd && index == n - 1) {
            __half* srcData1 = (__half*)srcData;
            __half* dstData1 = (__half*)dstData;
            __half value = __hfma(srcData1[index * 2], __float2half(scale), __float2half(shift));
            __half result = __float2half(1.0f);
            for (int i = 0; i < power; ++i) {
                result = __hmul(result, value);
            }
            dstData1[index * 2] = result;
        } else {
            __half2 value = __hfma2(srcData[index], __float2half2_rn(scale), __float2half2_rn(shift));
            __half2 result = __float2half2_rn(1.0f);
            for (int i = 0; i < power; ++i) {
               result = __hmul2(result, value);
            }
            dstData[index] = result;
        }
    }
#endif
}

Status CudaPowLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CudaPowLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaPowLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto params = dynamic_cast<PowLayerParam *>(param_);
    if (!params) {
        LOGE("Error: PowLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: PowLayerParam is nil");
    }

    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];
    int count = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims);
    if (input_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        float* input_data = static_cast<float*>(input_blob->GetHandle().base);
        float* output_data = static_cast<float*>(output_blob->GetHandle().base);

        pow_kernel<<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
            count, input_data, params->exponent, params->scale, params->shift, output_data);
    } else if (input_blob->GetBlobDesc().data_type == DATA_TYPE_HALF) {
        __half2* input_data = static_cast<__half2*>(input_blob->GetHandle().base);
        __half2* output_data = static_cast<__half2*>(output_blob->GetHandle().base);

        bool odd = count & 0x1;
        int thread_count = (count + 1) / 2;
        pow_kernel_fp16<<<TNN_CUDA_GET_BLOCKS(thread_count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
            thread_count, odd, input_data, params->exponent, params->scale, params->shift, output_data);
    } else {
        LOGE("Error: layer acc dont support datatype: %d\n", input_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: layer acc don't support datatype");
    }
    return TNN_OK;
}

REGISTER_CUDA_ACC(Pow, LAYER_POWER);

}  // namespace TNN_NS