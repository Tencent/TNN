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
#include "tnn/device/cuda/acc/cuda_pooling_1d_layer_acc.h"

namespace TNN_NS {

CudaPooling1DLayerAcc::~CudaPooling1DLayerAcc() {
    cudnnDestroyPoolingDescriptor(this->m_pooling_desc);
    cudnnDestroyTensorDescriptor(this->m_input_desc);
    cudnnDestroyTensorDescriptor(this->m_output_desc);
}

Status CudaPooling1DLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret = CudaLayerAcc::Init(context, param, resource, inputs, outputs);
    if (ret != TNN_OK) {
        return ret;
    }

    auto params = dynamic_cast<PoolingLayerParam *>(param);
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

    int kernels[2] = {params->kernels[0], 1};
    int padding[2] = {params->pads[0], 0};
    int strides[2] = {params->strides[0], 1};

    // cudnn don't support pool1d, add dimension 1 for using cudnnPooling
    cudnnSetPooling2dDescriptor(this->m_pooling_desc, this->m_pooling_mode, CUDNN_PROPAGATE_NAN,
        kernels[0], kernels[1], padding[0], padding[1], strides[0], strides[1]);
    
    return TNN_OK;
}

Status CudaPooling1DLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaPooling1DLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<PoolingLayerParam *>(param_);
    if (!param) {
        return Status(TNNERR_MODEL_ERR, "Error: PoolingLayerParam is nil");
    }

    Blob* input_blob = inputs[0];
    Blob* output_blob = outputs[0];
    float* input_data = static_cast<float*>(input_blob->GetHandle().base);
    float* output_data = static_cast<float*>(output_blob->GetHandle().base);

    auto input_dims = input_blob->GetBlobDesc().dims;
    auto output_dims = output_blob->GetBlobDesc().dims;

    cudnnSetTensor4dDescriptor(this->m_input_desc, this->m_tensor_format, this->m_data_type,
        input_dims[0], input_dims[1], input_dims[2], 1);
    cudnnSetTensor4dDescriptor(this->m_output_desc, this->m_tensor_format, this->m_data_type,
        output_dims[0], output_dims[1], output_dims[2], 1);

    float alpha = 1.f;
    float beta = 0.f;
    auto status = cudnnPoolingForward(context_->cudnn_handle_, this->m_pooling_desc, &alpha, m_input_desc,
        input_data, &beta, m_output_desc, output_data);

    if (status != CUDNN_STATUS_SUCCESS) {
        printf("error: %d\n", status);
    }

    return TNN_OK;
}

REGISTER_CUDA_ACC(Pooling1D, LAYER_POOLING_1D);

}  // namespace TNN_NS