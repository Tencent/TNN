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

#include <cub/cub.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_radix_sort.cuh>

#include "tnn/device/cuda/acc/cuda_layer_acc.h"
#include "tnn/device/cuda/utils.cuh"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_CUDA_ACC(LayerNorm, LAYER_LAYER_NORM);

// Special Float2 Structure for LayerNorm, to calculate sum and variance sum within one CUB Reduction Call.
// v1 for sum, v2 for variance sum.
struct LNFloat2 {
    float v1; float v2;
    __device__ __host__ inline LNFloat2(const float a, const float b) : v1(a), v2(b) {}
    __device__ __host__ inline LNFloat2() : v1(0.), v2(0.) {}
    __device__ __host__ inline LNFloat2(const float& other): v1(other), v2(other * other) {}
    __device__ __host__ inline LNFloat2(const __half& other): v1(float(other)), v2(float(other) * float(other)) {}
    __device__ __host__ inline LNFloat2 operator+(const LNFloat2 &other) { return {v1 + other.v1, v2 + other.v2}; }
    __device__ __host__ inline LNFloat2 &operator+=(const LNFloat2 &other) { v1 += other.v1; v2 += other.v2; return *this; }
};

struct LNFloat2CustomSum {
    template <typename T>
    CUB_RUNTIME_FUNCTION __device__ __host__ __forceinline__
    T operator()(const T &a, const T &b) const {
        return a + b;
    }
    
    CUB_RUNTIME_FUNCTION __device__ __host__ __forceinline__
    LNFloat2 operator()(const LNFloat2 &a, const LNFloat2 &b) const {
        return {a.v1 + b.v1, a.v2 + b.v2};
    }
};

// Step 1: Set offset for CUB reduce kernel if necessary
__global__ void ln_set_reduce_offset_kernel(int *offset, const int channels, const int channel_area) {
    CUDA_KERNEL_LOOP(index, channels+1) {
        offset[index] = index * channel_area;
    }
}

// Step 4: Calculate Output with scale, bias and calculated mean, var.
template<typename T>
__global__ void ln_mul_add_kernel(const T *input, T *output, const T *scale, const T *bias,
                                  const LNFloat2 *mean_var,
                                  const int count, const float eps) {
    int offset = blockIdx.y * blockDim.y + threadIdx.x;
    int total_offset = blockIdx.x * count + offset;
    if (offset < count) {
        const float* mean_var_float = reinterpret_cast<const float*>(mean_var);
        float mean = mean_var_float[blockIdx.x * 2 + 0] / float(count);
        float var  = mean_var_float[blockIdx.x * 2 + 1] / float(count) - mean * mean;
        var = 1.0 / sqrt(var + eps);
        float k = float(scale[offset]) * var;
        float b = - mean * k + float(bias[offset]);
        output[total_offset] = T(float(input[total_offset]) * k + b);
    }
}


Status CudaLayerNormLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret = CudaLayerAcc::Init(context, param, resource, inputs, outputs);
    if (ret != TNN_OK) {
        return ret;
    }

    // Create TempBuffer in Init Stage for LayerNorm
    CreateTempBuf(sizeof(LNFloat2) * 4); // Buffer 0 for Stored Mean & Var
    CreateTempBuf(sizeof(LNFloat2) * 4); // Buffer 1 for Cub::Reduce Offsets
    CreateTempBuf(sizeof(LNFloat2) * 4); // Buffer 2 for Cub::Reduce Tempspace

    return TNN_OK;
}

Status CudaLayerNormLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaLayerNormLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Blob *input_blob  = inputs[0];
    Blob *scale_blob  = inputs[1];
    Blob *bias_blob   = inputs[2];
    Blob *output_blob = outputs[0];

    auto layer_param = dynamic_cast<LayerNormLayerParam *>(param_);
    auto dims_input = input_blob->GetBlobDesc().dims;
    const int reduce_dim_size = layer_param->reduce_dims_size;

    if (layer_param->reduce_dims_size != scale_blob->GetBlobDesc().dims.size()) {
        return Status(TNNERR_PARAM_ERR, "LayerNormLayer has invalid dims for input blob of scale or bias");
    }

    const int channel_dim_size = (int)dims_input.size() - reduce_dim_size;
    const int channels = DimsVectorUtils::Count(dims_input, 0, channel_dim_size);
    const int channel_area = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims, channel_dim_size);
    if (0 == channels || 0 == channel_area) {
        LOGE("Error: blob count is zero\n");
        return Status(TNNERR_COMMON_ERROR, "Error: blob count is zero");
    }

    void *input_data  = input_blob->GetHandle().base;
    void *output_data = output_blob->GetHandle().base;
    void *scale_data  = scale_blob->GetHandle().base;
    void *bias_data   = bias_blob->GetHandle().base;

    const int THREAD_PER_BLOCK = 1024;
    int num_blocks = (channel_area - 1) / THREAD_PER_BLOCK + 1;
    dim3 griddim;
    griddim.x = channels; // batch_size
    griddim.y = num_blocks;

    // Re-Allocate Temp Buffer if size of existing one is not enough.
    ResizeTempBuf(0, sizeof(LNFloat2) * channels); // Buffer for stored mean & var
    ResizeTempBuf(1, sizeof(int) * (channels + 1)); // Buffer for temp offsets
    LNFloat2* temp0_ptr = static_cast<LNFloat2*>(tempbufs_[0].ptr);
    int* offsets_ptr = static_cast<int*>(tempbufs_[1].ptr);
    LNFloat2CustomSum tuple2_custom_sum;

    // Step 1: Set offsets for CUB Reduction kernel if necessary
    ln_set_reduce_offset_kernel<<<1, 256, 0, context_->GetStream()>>>(offsets_ptr, channels, channel_area);

    if (input_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        // Step 2: Determine temporary device storage requirements for CUB reduction, allocate if necessary
        size_t curr_cub_temp_bytes = 0;
        CubDebug(cub::DeviceSegmentedReduce::Reduce(nullptr, curr_cub_temp_bytes, (float*)input_data, temp0_ptr,
                                                    channels, offsets_ptr, offsets_ptr + 1, tuple2_custom_sum, LNFloat2(0), context_->GetStream()));
        ResizeTempBuf(2, curr_cub_temp_bytes); // Buffer for Cub TempSpace

        // Step 3: Call CUB Reduction for a second time, Run mean var sum-reduction
        CubDebug(cub::DeviceSegmentedReduce::Reduce(tempbufs_[2].ptr, curr_cub_temp_bytes, (float*)input_data, temp0_ptr,
                                                    channels, offsets_ptr, offsets_ptr + 1, tuple2_custom_sum, LNFloat2(0), context_->GetStream()));

        // Step 4: LayerNorm Multiple & Add with Reduced Mean & Var
        ln_mul_add_kernel<float><<<griddim, THREAD_PER_BLOCK, 0, context_->GetStream()>>>
            ((float*)input_data, (float *)output_data, (float *)scale_data, (float *)bias_data,
             temp0_ptr, channel_area, layer_param->eps);
    } else if (input_blob->GetBlobDesc().data_type == DATA_TYPE_HALF) {
        // Step 2: Determine temporary device storage requirements for CUB reduction, allocate if necessary
        size_t curr_cub_temp_bytes = 0;
        CubDebug(cub::DeviceSegmentedReduce::Reduce(nullptr, curr_cub_temp_bytes, (__half*)input_data, temp0_ptr,
                                                    channels, offsets_ptr, offsets_ptr + 1, tuple2_custom_sum, LNFloat2(0), context_->GetStream()));
        ResizeTempBuf(2, curr_cub_temp_bytes); // Buffer for Cub TempSpace

        // Step 3: Call CUB Reduction for a second time, Run mean var sum-reduction
        CubDebug(cub::DeviceSegmentedReduce::Reduce(tempbufs_[2].ptr, curr_cub_temp_bytes, (__half*)input_data, temp0_ptr,
                                                    channels, offsets_ptr, offsets_ptr + 1, tuple2_custom_sum, LNFloat2(0), context_->GetStream()));

        // Step 4: LayerNorm Multiple & Add with Reduced Mean & Var
        ln_mul_add_kernel<__half><<<griddim, THREAD_PER_BLOCK, 0, context_->GetStream()>>>
            ((__half*)input_data, (__half *)output_data, (__half *)scale_data, (__half *)bias_data,
             temp0_ptr, channel_area, layer_param->eps);
    } else {
        LOGE("Error: LayerNorm layer acc does not support datatype: %d\n", input_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: LayerNorm layer acc does not support current datatype");
    }

    return TNN_OK;
}

REGISTER_CUDA_ACC(LayerNorm, LAYER_LAYER_NORM);

}  // namespace TNN_NS
