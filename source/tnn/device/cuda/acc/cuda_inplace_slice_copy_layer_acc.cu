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

DECLARE_CUDA_ACC(InplaceSliceCopy, LAYER_INPLACE_SLICE_COPY);

template<typename T>
__global__ void strided_copy_kernel(const T* src, T* dst, const int count, const int offset,
    const int* strides_out, const int* strided_dims, const int* strides_in) 
{
    CUDA_KERNEL_LOOP(idx, count) {
        int d4 = idx / strides_in[4] % strided_dims[4];
        int d3 = idx / strides_in[3] % strided_dims[3];
        int d2 = idx / strides_in[2] % strided_dims[2];
        int d1 = idx / strides_in[1] % strided_dims[1];
        int d0 = idx / strides_in[0] % strided_dims[0];
        int index_in = d0 * strides_out[0] +
                       d1 * strides_out[1] +
                       d2 * strides_out[2] +
                       d3 * strides_out[3] +
                       d4 * strides_out[4] + offset;
        dst[index_in] = src[idx];
    }
}

template __global__ void strided_copy_kernel<float>(
    const float* src, float* dst, const int count, const int offset,
    const int* strides_fuse, const int* dims_out, const int* strides_out
);

template __global__ void strided_copy_kernel<half>(
    const half* src, half* dst, const int count, const int offset,
    const int* strides_fuse, const int* dims_out, const int* strides_out
);

Status CudaInplaceSliceCopyLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret = CudaLayerAcc::Init(context, param, resource, inputs, outputs);
    if (ret != TNN_OK) {
        return ret;
    }
    CreateTempBuf(5 * sizeof(int));
    CreateTempBuf(5 * sizeof(int));
    CreateTempBuf(5 * sizeof(int));

    return TNN_OK;
}

Status CudaInplaceSliceCopyLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaInplaceSliceCopyLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Blob *input_blob0   = inputs[0];
    Blob *input_blob1   = inputs[1];
    Blob *output_blob   = outputs[0];

    auto input0_dims    = input_blob0->GetBlobDesc().dims;
    auto input1_dims    = input_blob1->GetBlobDesc().dims;
    auto output_dims    = output_blob->GetBlobDesc().dims;
    auto layer_param    = dynamic_cast<StrideSliceV2LayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: InplaceSliceCopyLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: InplaceSliceCopyLayerParam is nil");
    }

    auto axes = layer_param->axes;
    std::vector<int> begins(5, 0), strides(5, 1), strided_dims(5, 1), strides_in(5, 1), strides_out(5, 1), strides_offset(5, 1);
    int offset = 0;
    for (int i = 0; i < axes.size(); ++i) {
        int axis = axes[i];
        int begin = layer_param->begins[i];
        begins[axis] = begin >= 0 ? begin : begin + output_dims[axis];
        strides[axis] = layer_param->strides[i];
    }
    for (int i = 0; i < input1_dims.size(); i++) {
        strided_dims[i] = input1_dims[i];
    }
    for (int i = input1_dims.size(); i < 5; i++) {
        strided_dims[i] = 1;
    }
    strides_in[4] = 1;
    strides_offset[4] = 1;
    for (int i = 3; i >= 0; i--) {
        if (i < input1_dims.size() - 1) {
            strides_in[i] = strides_in[i + 1] * input1_dims[i + 1];
            strides_offset[i] = strides_offset[i + 1] * output_dims[i + 1];
        } else {
            strides_in[i] = strides_in[i + 1];
            strides_offset[i] = strides_offset[i + 1];
        }
    }
    for (int i = 4; i >= 0; i--) {
        offset += begins[i] * strides_offset[i];
        strides_out[i] = strides_offset[i] * strides[i];
    }

    cudaMemcpyAsync(tempbufs_[0].ptr, &(strides_out[0]), 5 * sizeof(int), cudaMemcpyHostToDevice, context_->GetStream());
    cudaMemcpyAsync(tempbufs_[1].ptr, &(strided_dims[0]), 5 * sizeof(int), cudaMemcpyHostToDevice, context_->GetStream());
    cudaMemcpyAsync(tempbufs_[2].ptr, &(strides_in[0]), 5 * sizeof(int), cudaMemcpyHostToDevice, context_->GetStream());

    void* input0_ptr = input_blob0->GetHandle().base;
    void* input1_ptr = input_blob1->GetHandle().base;
    void* output_ptr = output_blob->GetHandle().base;

    int count_out = DimsVectorUtils::Count(output_dims);
    int count = DimsVectorUtils::Count(input1_dims);

    if (input_blob0->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        cudaMemcpyAsync(output_ptr, input0_ptr, count_out * sizeof(float), cudaMemcpyDeviceToDevice, context_->GetStream());
        if (count == 0)
            return TNN_OK;
        strided_copy_kernel<float><<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
            (float*)input1_ptr, (float*)output_ptr, count, offset, (const int*)tempbufs_[0].ptr,
            (const int*)tempbufs_[1].ptr, (const int*)tempbufs_[2].ptr
        );
    } else if (input_blob0->GetBlobDesc().data_type == DATA_TYPE_HALF) {
        cudaMemcpyAsync(output_ptr, input0_ptr, count_out * sizeof(half), cudaMemcpyDeviceToDevice, context_->GetStream());
        if (count == 0)
            return TNN_OK;
        strided_copy_kernel<half><<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
            (half*)input1_ptr, (half*)output_ptr, count, offset, (const int*)tempbufs_[0].ptr,
            (const int*)tempbufs_[1].ptr, (const int*)tempbufs_[2].ptr
        );
    } else {
        LOGE("Error: layer acc don't support data type: %d\n", inputs[0]->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: layer acc don't support datatype");
    }
    return TNN_OK;
}

REGISTER_CUDA_ACC(InplaceSliceCopy, LAYER_INPLACE_SLICE_COPY);

}