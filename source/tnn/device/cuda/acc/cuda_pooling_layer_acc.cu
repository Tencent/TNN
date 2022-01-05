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

#include "tnn/device/cuda/acc/cuda_pooling_layer_acc.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

__device__ int get_start_index(int a, int b, int c) {
    return (int)floorf((float)(a * c) / b);
}

__device__ int get_end_index(int a, int b, int c) {
    return (int)ceilf((float)((a + 1) * c) / b);
}

__global__ void adaptive_pooling_kernel(const float* input, float* output, int channels, int input_height,
        int input_width, int output_height, int output_width, int pool_type) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= output_height * output_width) return;

    int bid = blockIdx.y + blockIdx.z * gridDim.y;
    if (bid >= channels) return;

    int oh = tid / output_width;
    int ow = tid % output_width;

    int ih0 = get_start_index(oh, output_height, input_height);
    int ih1 = get_end_index(oh, output_height, input_height);
    int kh = ih1 - ih0;

    int iw0 = get_start_index(ow, output_width, input_width);
    int iw1 = get_end_index(ow, output_width, input_width);
    int kw = iw1 - iw0;

    const float* input_ptr = input + bid * input_height * input_width;
    float* output_ptr = output + bid * output_height * output_width;

    if (pool_type == 1) {
        float sum = 0;
        for (int ih = ih0; ih < ih1; ih++) {
            for (int iw = iw0; iw < iw1; iw++) {
                sum += input_ptr[ih * input_width + iw];
            }
        }
        output_ptr[oh * output_width + ow] = sum / kh / kw;
        }
}

CudaPoolingLayerAcc::~CudaPoolingLayerAcc() {
    cudnnDestroyPoolingDescriptor(this->m_pooling_desc);
    cudnnDestroyTensorDescriptor(this->m_input_desc);
    cudnnDestroyTensorDescriptor(this->m_output_desc);
}

Status CudaPoolingLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret = CudaLayerAcc::Init(context, param, resource, inputs, outputs);
    if (ret != TNN_OK) {
        return ret;
    }

    auto params = dynamic_cast<PoolingLayerParam*>(param);
    if (params->pool_type == 0) {
        this->m_pooling_mode = CUDNN_POOLING_MAX;
    } else {
        this->m_pooling_mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
    }

    this->m_tensor_format = CUDNN_TENSOR_NCHW;
    this->m_data_type = CUDNN_DATA_FLOAT;
    cudnnCreatePoolingDescriptor(&m_pooling_desc);
    cudnnCreateTensorDescriptor(&m_input_desc);
    cudnnCreateTensorDescriptor(&m_output_desc);
    auto input_dims = inputs[0]->GetBlobDesc().dims;
    auto output_dims = outputs[0]->GetBlobDesc().dims;
    cudnnSetPooling2dDescriptor(this->m_pooling_desc, this->m_pooling_mode, CUDNN_PROPAGATE_NAN,
        params->kernels[1], params->kernels[0], params->pads[2], params->pads[0], params->strides[1],
        params->strides[0]);
    return TNN_OK;
}

Status CudaPoolingLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaPoolingLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<PoolingLayerParam *>(param_);
    if (!param) {
        return Status(TNNERR_MODEL_ERR, "Error: PoolingLayerParam is nil");
    }

    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];
    float* input_data = static_cast<float*>(input_blob->GetHandle().base);
    float* output_data = static_cast<float*>(output_blob->GetHandle().base);

    auto input_dims = input_blob->GetBlobDesc().dims;
    auto output_dims = output_blob->GetBlobDesc().dims;
    if (param->is_global_pool) {
        cudnnSetPooling2dDescriptor(this->m_pooling_desc, this->m_pooling_mode, CUDNN_PROPAGATE_NAN,
            input_dims[2], input_dims[3], param->pads[2], param->pads[0], param->strides[1],
            param->strides[0]);
    }
    cudnnSetTensor4dDescriptor(this->m_input_desc, this->m_tensor_format, this->m_data_type,
        input_dims[0], input_dims[1], input_dims[2], input_dims[3]);
    cudnnSetTensor4dDescriptor(this->m_output_desc, this->m_tensor_format, this->m_data_type,
        output_dims[0], output_dims[1], output_dims[2], output_dims[3]);

    if (param->is_adaptive_pool) {
        bool is_1d = input_dims.size() == 3;
        int channels = is_1d ? input_dims[0] : input_dims[0] * input_dims[1];
        int input_height = is_1d ? input_dims[1] : input_dims[2];
        int input_width = is_1d ? input_dims[2] : input_dims[3];
        int output_height = is_1d ? output_dims[1] : output_dims[2];
        int output_width = is_1d ? output_dims[2] : output_dims[3];
        int count = output_height*output_width;

        dim3 grid(TNN_CUDA_GET_BLOCKS(count), std::min(channels, TNN_CUDA_MAX_GRID_DIM), (channels + TNN_CUDA_MAX_GRID_DIM - 1) / TNN_CUDA_MAX_GRID_DIM);
        adaptive_pooling_kernel<<<grid, TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
            input_data, output_data, channels, input_height, input_width, output_height, output_width,
            param->pool_type);
            
        auto error = cudaGetLastError();
        if (error != cudaSuccess) {
            LOGE("Error: pooling kernel error!\n %s\n", cudaGetErrorString(error));
            return Status(TNNERR_CUDA_KERNEL_LAUNCH_ERROR, "Error: pooling kernel error!");
        }
    } else {
        float alpha = 1.f;
        float beta = 0.f;
        cudnnPoolingForward(context_->cudnn_handle_, this->m_pooling_desc, &alpha, m_input_desc,
            input_data, &beta, m_output_desc, output_data);
    }
    return TNN_OK;
}

REGISTER_CUDA_ACC(Pooling, LAYER_POOLING);

}  // namespace TNN_NS

