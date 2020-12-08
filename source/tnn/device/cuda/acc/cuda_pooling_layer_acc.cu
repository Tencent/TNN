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
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

CudaPoolingLayerAcc::~CudaPoolingLayerAcc() {
    cudnnDestroy(this->m_cudnn_handle);
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
    cudnnCreate(&m_cudnn_handle);
    cudnnCreatePoolingDescriptor(&m_pooling_desc);
    cudnnCreateTensorDescriptor(&m_input_desc);
    cudnnCreateTensorDescriptor(&m_output_desc);

    cudnnSetPooling2dDescriptor(this->m_pooling_desc, this->m_pooling_mode, CUDNN_PROPAGATE_NAN,
        params->kernels[1], params->kernels[0], params->pads[2], params->pads[0], params->strides[1],
        params->strides[0]);
    auto input_dims = inputs[0]->GetBlobDesc().dims;
    auto output_dims = outputs[0]->GetBlobDesc().dims;
    cudnnSetTensor4dDescriptor(this->m_input_desc, this->m_tensor_format, this->m_data_type,
        input_dims[0], input_dims[1], input_dims[2], input_dims[3]);
    cudnnSetTensor4dDescriptor(this->m_output_desc, this->m_tensor_format, this->m_data_type,
        output_dims[0], output_dims[1], output_dims[2], output_dims[3]);
    return TNN_OK;
}

Status CudaPoolingLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaPoolingLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];
    float* input_data = static_cast<float*>(input_blob->GetHandle().base);
    float* output_data = static_cast<float*>(output_blob->GetHandle().base);
    float alpha = 1.f;
    float beta = 0.f;
    cudnnPoolingForward(this->m_cudnn_handle, this->m_pooling_desc, &alpha, m_input_desc,
        input_data, &beta, m_output_desc, output_data);
    return TNN_OK;
}

REGISTER_CUDA_ACC(Pooling, LAYER_POOLING);

}  // namespace TNN_NS

