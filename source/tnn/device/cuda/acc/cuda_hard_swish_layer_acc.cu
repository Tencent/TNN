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
#include "tnn/device/cuda/fastdiv.h"
#include "tnn/utils/dims_utils.h"
#include <cuda.h>

namespace TNN_NS {

DECLARE_CUDA_ACC(HardSwish, LAYER_HARDSWISH);

template <typename T>
__device__ __forceinline__ float toFloat(T x)
{
    return float(x);
}

template <typename T>
__device__ T fromFloat(float x);

template <>
__device__ __forceinline__ int8_t fromFloat<int8_t>(float x)
{
    // The order of the next two statements matters when x is a NaN,
    // because IEEE max/min return the non-NaN operand when one operand
    // is a NaN and the other is not.
    x = fmaxf(x, INT8_MIN);
    x = fminf(x, INT8_MAX);
    return __float2int_rn(x);
}

template <>
__device__ __forceinline__ float fromFloat<float>(float x)
{
    return x;
}

template <>
__device__ __forceinline__ __half fromFloat<__half>(float x)
{
    return __float2half(x);
}

template <>
__device__ __forceinline__ int32_t fromFloat<int32_t>(float x)
{
    return __float2int_rz(x);
}

template <typename T, bool packed, typename T_MATH, bool isI8, int32_t BlockDim, int32_t PACK, int32_t UNROLL>
__global__ __launch_bounds__(BlockDim) void hardswish_kernel(T *dst, int32_t dstNStride, const T *src,
                                                             int32_t srcNStride, const float i0i8Scale,
                                                             const float o0i8Scale, const float alpha, const float beta,
                                                             int32_t N, int32_t C, int32_t H, int32_t W, fastdiv divCHW,
                                                             fastdiv divHW, fastdiv divW, int32_t cExtent) {
    typedef int32_t int_cast_type;
    alignas(alignof(int_cast_type)) T data_in[UNROLL];
    alignas(alignof(int_cast_type)) T data_out[UNROLL];

    int32_t tid = (blockIdx.x * BlockDim + threadIdx.x) * UNROLL;
    int32_t n   = tid / divCHW;
    int32_t nr0 = tid % divCHW;
    if (n < N) {
        int32_t index                               = n * srcNStride + nr0;
        int32_t dndex                               = n * dstNStride + nr0;
        *reinterpret_cast<int_cast_type *>(data_in) = *reinterpret_cast<int_cast_type *>((T *)src + index);

#pragma unroll UNROLL
        for (int32_t i = 0; i < UNROLL; i++) {
            T_MATH input = toFloat(data_in[i]);
            T_MATH output;
            int32_t nr  = nr0 + i;
            int32_t idx = nr;

            int32_t c      = nr / divHW;
            idx            = nr % divHW;
            int32_t w      = idx % divW;
            int32_t offset = (packed) ? c * PACK + (w & (PACK - 1)) : c;
            if (offset >= cExtent) {
                data_out[i] = fromFloat<T>(0.f);  // ensure correct zero-padding
                continue;
            }
            if (isI8) {
                float i8Scale = i0i8Scale;
                input *= i8Scale;
            }

            // x * clip(x*alpha + beta, 0, 1)
            output = input * max(min(input * alpha + beta, 1.f), 0.f);

            if (isI8) {
                float o8Scale = o0i8Scale;
                output /= o8Scale;
            }
            data_out[i] = fromFloat<T>(output);
        }

        *reinterpret_cast<int_cast_type *>(dst + dndex) = *reinterpret_cast<int_cast_type *>(data_out);
    }
}

// template<typename T>
// __global__ void hard_swish_elementwise_kernel(const int count, const T *input, T *output, const float alpha, const float beta) {
//     CUDA_KERNEL_LOOP(index, count) {
//         output[index] = input[index] * T(max(min(float(input[index]) * alpha + beta, 1.f), 0.f));
//     }
// }

// template<typename T>
// __global__ void hard_swish_kernel(int count, const T* in1, const T* in2, T* out, int in_n1,
//         int in_c1, int in_h1, int in_w1, int in_n2, int in_c2, int in_h2, int in_w2, int out_c, int out_h,
//         int out_w, const float alpha, const float beta) {
//     CUDA_KERNEL_LOOP(index, count) {
//         int b = index / (out_c * out_h * out_w);
//         int c = index / (out_h * out_w) % out_c;
//         int h = index / out_w % out_h;
//         int w = index % out_w;
//         int input_index_b_1 = min(b, in_n1-1) * in_c1 * in_h1 * in_w1;
//         int input_index_b_2 = min(b, in_n2-1) * in_c2 * in_h2 * in_w2;
//         int input_index_c_1 = min(c, in_c1-1) * in_h1 * in_w1 + input_index_b_1;
//         int input_index_c_2 = min(c, in_c2-1) * in_h2 * in_w2 + input_index_b_2;
//         int input_index_h_1 = min(h, in_h1-1) * in_w1 + input_index_c_1;
//         int input_index_h_2 = min(h, in_h2-1) * in_w1 + input_index_c_2;
//         int input_index_w_1 = min(w, in_w1-1) + input_index_h_1;
//         int input_index_w_2 = min(w, in_w2-1) + input_index_h_2;
//         out[index] = in1[input_index_w_1] * T(max(min(float(in2[input_index_w_2]) * alpha + beta, 1.f), 0.f));
//     }
// }

Status CudaHardSwishLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CudaHardSwishLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaHardSwishLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto params = dynamic_cast<HardSwishLayerParam *>(param_);
    if (!params) {
        LOGE("Error: HardSwishLayerParam is nil\n");
        return Status(TNNERR_LAYER_ERR, "Error: HardSwishLayerParam is nil");
    }

    auto cuda_context = dynamic_cast<CudaContext *>(context_);

    auto input_blob  = inputs[0];
    auto output_blob = outputs[0];

    void *input_data  = static_cast<void *>(input_blob->GetHandle().base);
    void *output_data = static_cast<void *>(output_blob->GetHandle().base);

    auto dtype      = input_blob->GetBlobDesc().data_type;
    auto dformat    = input_blob->GetBlobDesc().data_format;
    auto input_dims = input_blob->GetBlobDesc().dims;
    int N           = DimsFunctionUtils::GetDim(input_dims, 0);
    int C           = DimsFunctionUtils::GetDim(input_dims, 1);
    int H           = DimsFunctionUtils::GetDim(input_dims, 2);
    int W           = DimsFunctionUtils::GetDim(input_dims, 3);
    int c_extend    = C;
    const int block = 128;

    if (dtype == DATA_TYPE_INT8) {
        auto input_scale_handle  = cuda_context->GetQuantResource(input_blob->GetBlobDesc().name);
        auto output_scale_handle = cuda_context->GetQuantResource(output_blob->GetBlobDesc().name);

        auto iscale = input_scale_handle->force_to<float *>()[0];
        auto oscale = output_scale_handle->force_to<float *>()[0];
        // float iscale = 0.05f;
        // float oscale = 0.05f;

        if (dformat == DATA_FORMAT_NC4HW4) {
            W              = W * 4;
            C              = UP_DIV(C, 4);
            const int size = N * C * H * W;
            fastdiv divCHW;
            divCHW.init(C * H * W);
            fastdiv divHW;
            divHW.init(H * W);
            fastdiv divW;
            divW.init(W);
            const int srcNStride = C * H * W;
            const int dstNStride = srcNStride;
            hardswish_kernel<int8_t, true, float, true, block, 4, 4>
                <<<UP_DIV(size, block * 4), block, 0, context_->GetStream()>>>(
                    (int8_t *)output_data, dstNStride, (int8_t *)input_data, srcNStride, iscale, oscale, params->alpha,
                    params->beta, N, C, H, W, divCHW, divHW, divW, c_extend);
        } else if (dformat == DATA_FORMAT_NC32HW32) {
            W              = W * 32;
            C              = UP_DIV(C, 32);
            const int size = N * C * H * W;
            fastdiv divCHW;
            divCHW.init(C * H * W);
            fastdiv divHW;
            divHW.init(H * W);
            fastdiv divW;
            divW.init(W);
            const int srcNStride = C * H * W;
            const int dstNStride = srcNStride;
            hardswish_kernel<int8_t, true, float, true, block, 32, 4>
                <<<UP_DIV(size, block * 4), block, 0, context_->GetStream()>>>(
                    (int8_t *)output_data, dstNStride, (int8_t *)input_data, srcNStride, iscale, oscale, params->alpha,
                    params->beta, N, C, H, W, divCHW, divHW, divW, c_extend);
        } else {
            LOGE("Error: unsupported int8 layout\n");
            return Status(TNNERR_LAYER_ERR, "Error: unsupported int8 layout");
        }
    } else {
        LOGE("Error: unsupported data type\n");
        return Status(TNNERR_LAYER_ERR, "Error: unsupported data type");
    }

    // int count = DimsVectorUtils::Count(outputs[0]->GetBlobDesc().dims);

    // Blob* input_blob1 = inputs[0];
    // Blob* input_blob2 = inputs[0];
    // Blob* output_blob = outputs[0];
    // if (inputs.size() != 1) {
    //     input_blob2 = inputs[1];
    // } else {

    //     int count = DimsVectorUtils::Count(input_blob1->GetBlobDesc().dims);
    //     if (input_blob1->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
    //         float* input_data = static_cast<float*>(input_blob1->GetHandle().base);
    //         float* output_data = static_cast<float*>(output_blob->GetHandle().base);
    //         hard_swish_elementwise_kernel<float><<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
    //             count, input_data, output_data, params->alpha, params->beta
    //         );
    //     } else if (input_blob1->GetBlobDesc().data_type == DATA_TYPE_HALF) {
    //         printf("half here\n");
    //         half* input_data = static_cast<half*>(input_blob1->GetHandle().base);
    //         half* output_data = static_cast<half*>(output_blob->GetHandle().base);
    //         hard_swish_elementwise_kernel<half><<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
    //             count, input_data, output_data, params->alpha, params->beta
    //         );
    //     }

    //     // auto error = cudaGetLastError();
    //     // if (error != cudaSuccess) {
    //     //     LOGE("Error: hard swish kernel error!\n %s\n", cudaGetErrorString(error));
    //     //     return Status(TNNERR_CUDA_KERNEL_LAUNCH_ERROR, "Error: hard swish kernel error!");
    //     // }
    //     return TNN_OK;
    // }

    // auto input_dims1 = input_blob1->GetBlobDesc().dims;
    // auto input_dims2 = input_blob2->GetBlobDesc().dims;
    // auto output_dims = output_blob->GetBlobDesc().dims;

    // int in_n1 = DimsFunctionUtils::GetDim(input_dims1, 0);
    // int in_c1 = DimsFunctionUtils::GetDim(input_dims1, 1);
    // int in_h1 = DimsFunctionUtils::GetDim(input_dims1, 2);
    // int in_w1 = DimsFunctionUtils::GetDim(input_dims1, 3);

    // int in_n2 = DimsFunctionUtils::GetDim(input_dims2, 0);
    // int in_c2 = DimsFunctionUtils::GetDim(input_dims2, 1);
    // int in_h2 = DimsFunctionUtils::GetDim(input_dims2, 2);
    // int in_w2 = DimsFunctionUtils::GetDim(input_dims2, 3);

    // int out_c = DimsFunctionUtils::GetDim(output_dims, 1);
    // int out_h = DimsFunctionUtils::GetDim(output_dims, 2);
    // int out_w = DimsFunctionUtils::GetDim(output_dims, 3);

    // if (input_blob1->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
    //     float* input_data1 = static_cast<float*>(input_blob1->GetHandle().base);
    //     float* input_data2 = static_cast<float*>(input_blob2->GetHandle().base);
    //     float* output_data = static_cast<float*>(output_blob->GetHandle().base);
        
    //     hard_swish_kernel<<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
    //         count, input_data1, input_data2, output_data, in_n1, in_c1, in_h1, in_w1, in_n2, in_c2, in_h2,
    //         in_w2, out_c, out_h, out_w, params->alpha, params->beta);
    // } else if (input_blob1->GetBlobDesc().data_type == DATA_TYPE_HALF) {
    //     half* input_data1 = static_cast<half*>(input_blob1->GetHandle().base);
    //     half* input_data2 = static_cast<half*>(input_blob2->GetHandle().base);
    //     half* output_data = static_cast<half*>(output_blob->GetHandle().base);
    //     hard_swish_kernel<half><<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
    //         count, input_data1, input_data2, output_data, in_n1, in_c1, in_h1, in_w1, in_n2, in_c2, in_h2,
    //         in_w2, out_c, out_h, out_w, params->alpha, params->beta);
    // }
    
    return TNN_OK;
}

REGISTER_CUDA_ACC(HardSwish, LAYER_HARDSWISH);

} // namespace TNN_NS
