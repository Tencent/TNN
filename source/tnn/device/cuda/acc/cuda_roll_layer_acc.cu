// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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
#include "tnn/device/cuda/cuda_macro.h"

namespace TNN_NS {

DECLARE_CUDA_ACC(Roll, LAYER_ROLL);

Status CudaRollLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);;
}

Status CudaRollLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

typedef struct roll_dims_t {
    roll_dims_t(std::vector<int> dims) {
        memset(d, 0, maxDims * sizeof(int));
        nbDims = dims.size();
        for (int i=0; i<nbDims; i++) {
            d[i] = dims[i];
        }
    }
    constexpr static int maxDims = 6;
    int nbDims = 0;
    int d[maxDims];
} dims_t;

template<typename T, int NBDIMS>
__global__ void roll_kernel(const T* src, T* dst,
                            const dims_t shape, const dims_t shifts, const dims_t strides, const int count) {
    CUDA_KERNEL_LOOP(index, count) {
        int src_index = 0;
        int dst_index = index;
        int remainder = index;
        //#pragma unroll
        for (int dim=0; dim<NBDIMS; dim++) {
            int dst_index_dim = remainder / strides.d[dim];
            int src_index_dim  = (dst_index_dim - shifts.d[dim] + shape.d[dim]) % shape.d[dim];
            src_index += strides.d[dim] * src_index_dim;
            remainder %= strides.d[dim];
        }

        dst[dst_index] = src[src_index];
    }
}

Status CudaRollLayerAcc::Forward(const std::vector<Blob*> &inputs, const std::vector<Blob*> &outputs) {
    // Operator Roll input.dim == output.dim
    auto input_blob  = inputs[0];
    auto output_blob = outputs[0];
    auto input_dims  = input_blob->GetBlobDesc().dims;
    auto count       = DimsVectorUtils::Count(input_dims);

    auto roll_param  = dynamic_cast<RollLayerParam*>(param_);
    if (roll_param == nullptr) {
        LOGE("Error: CudaRollLayer forward load layer param failed\n");
        return Status(TNNERR_MODEL_ERR, "Error: CudaRollLayer forward Load layer param failed!");
    }
    if (roll_param->dims.size() != roll_param->shifts.size()) {
        LOGE("Error: CpuRollLayer forward layer param.shifts.nbDims not equal to input param.dims.nbDims.\n");
        return Status(TNNERR_MODEL_ERR, "Error: CpuRollLayer forward layer param.shifts.nbDims not equal to input param.dims.nbDims!");
    }

    // Create Full, Ordered, Positive shifts from param.shifts.
    // Create Strides.
    std::vector<int> shifts(input_dims.size(), 0);
    std::vector<int> strides(input_dims.size(), 1);
    for (int d=0; d<input_dims.size(); d++) {
        strides[d] = DimsVectorUtils::Count(input_dims, d+1);
    }
    for (int d=0; d<roll_param->dims.size(); d++) {
        int dim     = roll_param->dims[d];
        shifts[dim] = roll_param->shifts[d] < 0 ? roll_param->shifts[d] + input_dims[dim] : roll_param->shifts[d];
    }
    dims_t shape_dims(input_dims);
    dims_t shifts_dims(shifts);
    dims_t strides_dims(strides);
    
    const int THREAD_PER_BLOCK = 128;

    dim3 blocks;
    blocks.x = (count + THREAD_PER_BLOCK - 1 ) / THREAD_PER_BLOCK;
    if (blocks.x > 65535) {
        LOGE("Error: CudaRollLayer forward layer cuda block.x > 65535, large kernel not supported yet.\n");
        return Status(TNNERR_MODEL_ERR, "Error: CudaRollLayer forward layer cuda block.x > 65535, large kernel not supported yet.");
    }

    // Run cuda Kernel
    auto data_type = outputs[0]->GetBlobDesc().data_type;
    if (data_type == DATA_TYPE_FLOAT || data_type == DATA_TYPE_INT32) {
        // DataType with Element size = 4, treat as float
        float* src = reinterpret_cast<float*>(((char*)input_blob->GetHandle().base) + input_blob->GetHandle().bytes_offset);
        float* dst = reinterpret_cast<float*>(((char*)output_blob->GetHandle().base) + output_blob->GetHandle().bytes_offset);
        
        using kernel_function_ptr_t = decltype(&roll_kernel<float,1>);
        kernel_function_ptr_t kernel_ptr = nullptr;
        switch (shape_dims.nbDims) {
            case 1:
                kernel_ptr = roll_kernel<float, 1>;
                break;
            case 2:
                kernel_ptr = roll_kernel<float, 2>;
                break;
            case 3:
                kernel_ptr = roll_kernel<float, 3>;
                break;
            case 4:
                kernel_ptr = roll_kernel<float, 4>;
                break;
            case 5:
                kernel_ptr = roll_kernel<float, 5>;
                break;
            case 6:
                kernel_ptr = roll_kernel<float, 6>;
                break;
            default:
                LOGE("Error: CudaRollLayer forward layer input nbDims should be 1-6.\n");
                return Status(TNNERR_MODEL_ERR, "Error: CudaRollLayer forward layer input nbDims should be 1-6.");
        }
        kernel_ptr<<<blocks, THREAD_PER_BLOCK, 0, context_->GetStream()>>>
                    (src, dst, shape_dims, shifts_dims, strides_dims, count);
    } else if (data_type == DATA_TYPE_HALF || data_type == DATA_TYPE_BFP16) {
        // DataType with Element size = 2, treat as __half
        __half* src = reinterpret_cast<__half*>(((char*)input_blob->GetHandle().base) + input_blob->GetHandle().bytes_offset);
        __half* dst = reinterpret_cast<__half*>(((char*)output_blob->GetHandle().base) + output_blob->GetHandle().bytes_offset);
        using kernel_function_ptr_t = decltype(&roll_kernel<__half,1>);
        kernel_function_ptr_t kernel_ptr = nullptr;
        switch (shape_dims.nbDims) {
            case 1:
                kernel_ptr = roll_kernel<__half, 1>;
                break;
            case 2:
                kernel_ptr = roll_kernel<__half, 2>;
                break;
            case 3:
                kernel_ptr = roll_kernel<__half, 3>;
                break;
            case 4:
                kernel_ptr = roll_kernel<__half, 4>;
                break;
            case 5:
                kernel_ptr = roll_kernel<__half, 5>;
                break;
            case 6:
                kernel_ptr = roll_kernel<__half, 6>;
                break;
            default:
                LOGE("Error: CudaRollLayer forward layer input nbDims should be 1-6.\n");
                return Status(TNNERR_MODEL_ERR, "Error: CudaRollLayer forward layer input nbDims should be 1-6.");
        }
        kernel_ptr<<<blocks, THREAD_PER_BLOCK, 0, context_->GetStream()>>>
                    (src, dst, shape_dims, shifts_dims, strides_dims, count);
    } else {
        LOGE("Error: CudaRollLayer forward layer data type not supported.\n");
        return Status(TNNERR_MODEL_ERR, "Error: CudaRollLayer forward layer data type not supported!");
    }

    return TNN_OK;
}

REGISTER_CUDA_ACC(Roll, LAYER_ROLL);

}  // namespace TNN_NS
