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

#include <memory>

#include "tnn/device/cuda/acc/cuda_conv_layer_acc.h"
#include "tnn/device/cuda/acc/cuda_layer_acc.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

using perf_t = cudnnConvolutionFwdAlgoPerf_t;

Status CudaConvLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    CudaLayerAcc::Init(context, param, resource, inputs, outputs);
    alpha_          = 1.0f;
    beta_           = 0.0f;
    workspace_data_ = nullptr;
    workspace_size_ = 0;
    weights_        = nullptr;
    bias_           = nullptr;
    bias_term_      = false;

    DimsVector input_dims  = inputs[0]->GetBlobDesc().dims;
    DimsVector output_dims = outputs[0]->GetBlobDesc().dims;

    if (input_dims.size() == 0 || output_dims.size() == 0) {
        return TNNERR_LAYER_ERR;
    }

    Blob *input = inputs[0];

    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param);

    // only some 7x7 conv case need run with cudnn because trt fp16 bug
    bool symmetric = (conv_param->pads[0] == conv_param->pads[1]) && (conv_param->pads[2] == conv_param->pads[3]);
    if (!symmetric ||
        !((conv_param->kernels[1] == 7 && conv_param->kernels[0] == 7) ||
          (conv_param->kernels[1] == 41 && conv_param->kernels[0] == 1) ||
          (conv_param->kernels[1] == 5 && conv_param->kernels[0] == 1))) return TNN_OK;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bottom_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&top_desc_));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc_));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc_));
    CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc_, conv_param->group));

    const int filter_dims[] = {
        output_dims[1], input_dims[1] / conv_param->group,
        conv_param->kernels[1], conv_param->kernels[0]};

    CUDNN_CHECK(cudnnSetFilterNdDescriptor(filter_desc_, CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW, 4, filter_dims));

    const int pad_dims[] = {conv_param->pads[2], conv_param->pads[0]};
    const int sti_dims[] = {conv_param->strides[1], conv_param->strides[0]};
    const int dil_dims[] = {conv_param->dialations[1], conv_param->dialations[0]};

    CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(
        conv_desc_, 2, pad_dims, sti_dims, dil_dims, CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));

    ConvLayerResource *conv_resource =
        dynamic_cast<ConvLayerResource *>(resource);
    float *weights = conv_resource->filter_handle.force_to<float *>();

    size_t weights_size = sizeof(float) * input_dims[1] * output_dims[1] *
        conv_param->kernels[1] * conv_param->kernels[0];

    CUDA_CHECK(cudaMalloc((void **)&weights_, weights_size));
    CUDA_CHECK(cudaMemcpy(weights_, weights, weights_size, cudaMemcpyHostToDevice));

    if (conv_param->bias) {
        bias_term_ = true;
        if (output_dims[1] * sizeof(float) != conv_resource->bias_handle.GetBytesSize()) {
            return TNNERR_MODEL_ERR;
        }

        const int bias_dim[] = {1, output_dims[1], 1, 1};
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc_));
        CUDNN_CHECK(cudnnSetTensorNdDescriptorEx(
            bias_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 4, bias_dim));

        CUDA_CHECK(cudaMalloc((void **)&bias_, conv_resource->bias_handle.GetBytesSize()));
        CUDA_CHECK(cudaMemcpy(bias_, conv_resource->bias_handle.force_to<float *>(),
            conv_resource->bias_handle.GetBytesSize(), cudaMemcpyHostToDevice));
    }

    return this->Reshape(inputs, outputs);
}

CudaConvLayerAcc::~CudaConvLayerAcc(){
    if (workspace_data_ != nullptr) {
        CUDA_CHECK(cudaFree(workspace_data_));
    }
    if (weights_ != nullptr) {
        CUDA_CHECK(cudaFree(weights_));
    }
    if (bias_ != nullptr) {
        CUDA_CHECK(cudaFree(bias_));
    }
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bottom_desc_));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_desc_));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc_));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc_));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc_));
}

Status CudaConvLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);

    // only some 7x7 conv case need run with cudnn because trt fp16 bug
    bool symmetric = (conv_param->pads[0] == conv_param->pads[1]) && (conv_param->pads[2] == conv_param->pads[3]);
    if (!symmetric ||
        !((conv_param->kernels[1] == 7 && conv_param->kernels[0] == 7) ||
          (conv_param->kernels[1] == 41 && conv_param->kernels[0] == 1) ||
          (conv_param->kernels[1] == 5 && conv_param->kernels[0] == 1))) return TNN_OK;

    DimsVector input_dims  = inputs[0]->GetBlobDesc().dims;
    DimsVector output_dims = outputs[0]->GetBlobDesc().dims;

    int in_dims[] = {input_dims[0], input_dims[1], input_dims[2], input_dims[3]};
    CUDNN_CHECK(cudnnSetTensorNdDescriptorEx(bottom_desc_, CUDNN_TENSOR_NCHW,
                                             CUDNN_DATA_FLOAT, 4, in_dims));

    int out_dims[4];
    CUDNN_CHECK(cudnnGetConvolutionNdForwardOutputDim(
        conv_desc_, bottom_desc_, filter_desc_, 4, out_dims));

    CUDNN_CHECK(cudnnSetTensorNdDescriptorEx(top_desc_, CUDNN_TENSOR_NCHW,
                                             CUDNN_DATA_FLOAT, 4, out_dims));

    for(int i = 0; i < 4; i++) {
        if (out_dims[i] != output_dims[i]) {
            LOGE("CUDNN got different output shapes from TNN\n");
            return TNNERR_LAYER_ERR;
        }
    }

    // algorithm
    static constexpr int num_algos = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    int perf_count;
    std::unique_ptr<perf_t[]> perf_results(new perf_t[num_algos]);

    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
        context_->cudnn_handle_, bottom_desc_, filter_desc_, conv_desc_,
        top_desc_, num_algos, &perf_count, perf_results.get()));

    std::vector<perf_t> valid_algos;
    valid_algos.reserve(perf_count);
    for (int i = 0; i < perf_count; i++) {
        if (perf_results.get()[i].status == CUDNN_STATUS_SUCCESS) {
            valid_algos.push_back(perf_results.get()[i]);
        }
    }

    if (valid_algos.size() == 0) {
        LOGE("CUDNN get conv algo failed.\n");
        return TNNERR_LAYER_ERR;
    }
    conv_algo_ = valid_algos[0].algo;

    // workspace
    size_t needed_workspace_size;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        context_->cudnn_handle_, bottom_desc_, filter_desc_, conv_desc_,
        top_desc_, conv_algo_, &needed_workspace_size));

    if (workspace_size_ < needed_workspace_size) {
        workspace_size_ = needed_workspace_size;
        if (workspace_data_ != nullptr) {
            CUDA_CHECK(cudaFree(workspace_data_));
        }
        CUDA_CHECK(cudaMalloc(&workspace_data_, workspace_size_));
    }
    return TNN_OK;
}

Status CudaConvLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    CUDNN_CHECK(cudnnConvolutionForward(
        context_->cudnn_handle_, &alpha_, bottom_desc_,
        inputs[0]->GetHandle().base, filter_desc_, weights_, conv_desc_,
        conv_algo_, workspace_data_, workspace_size_, &beta_, top_desc_,
        outputs[0]->GetHandle().base));

    if (bias_term_) {
        float alpha = 1.0f;
        float beta  = 1.0f;
        CUDNN_CHECK(cudnnAddTensor(context_->cudnn_handle_, &alpha, bias_desc_,
                                   bias_, &beta, top_desc_,
                                   outputs[0]->GetHandle().base));
    }
    return TNN_OK;
}

REGISTER_CUDA_ACC(Conv, LAYER_CONVOLUTION);

}  // namespace TNN_NS

