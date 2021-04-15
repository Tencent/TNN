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

#include "tnn/device/cuda/acc/cuda_conv_3d_layer_acc.h"

#include <memory>

#include "tnn/device/cuda/acc/cuda_layer_acc.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

using perf_t = cudnnConvolutionFwdAlgoPerf_t;

std::vector<perf_t> getValidAlgorithms(perf_t *perfResults, int n_algo) {

    std::vector<perf_t> valid_results;
    valid_results.reserve(n_algo);
    for (int i = 0; i < n_algo; i++) {
        if (perfResults[i].status == CUDNN_STATUS_SUCCESS) {
            valid_results.push_back(perfResults[i]);
        }
    }
    return valid_results;
}


Status CudaConv3DLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
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

    Blob *input = inputs[0];

    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param);

    // LOGD("CudaConv3DLayer param kernel_w: %d, kernel_h: %d \n",
    // conv_param->kernels[0], conv_param->kernels[1]);

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bottom_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&top_desc_));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc_));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc_));
    CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc_, conv_param->group));

    const int filter_dims[] = {
        output_dims[1], input_dims[1] / conv_param->group,
        conv_param->kernels[2], conv_param->kernels[1], conv_param->kernels[0]};

    CUDNN_CHECK(cudnnSetFilterNdDescriptor(filter_desc_, CUDNN_DATA_FLOAT,
                                           CUDNN_TENSOR_NCHW, 5, filter_dims));

    const int pad_dims[] = {conv_param->pads[4], conv_param->pads[2],
                            conv_param->pads[0]};  // DHW
    const int sti_dims[] = {conv_param->strides[2], conv_param->strides[1],
                            conv_param->strides[0]};
    const int dil_dims[] = {conv_param->dialations[2], conv_param->dialations[1],
                            conv_param->dialations[1]};

    CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(
        conv_desc_, 3, pad_dims, sti_dims, dil_dims, CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));

    ConvLayerResource *conv_resource =
        dynamic_cast<ConvLayerResource *>(resource);
    float *weights = conv_resource->filter_handle.force_to<float *>();
    // LOGD("weight size: %d \n", conv_resource->filter_handle.GetBytesSize());
    // LOGD("weights0: %f \n", weights[0]);

    size_t weights_size = sizeof(float) * input_dims[1] *
                          output_dims[1] * conv_param->kernels[2] *
                          conv_param->kernels[1] * conv_param->kernels[0];

    CUDA_CHECK(cudaMalloc((void **)&weights_, weights_size));
    CUDA_CHECK(cudaMemcpy(weights_, weights, weights_size, cudaMemcpyHostToDevice));

    // LOGD("CudaConv3DLayer bias: %d \n", conv_param->bias);
    // LOGD("CudaConv3DLayer bias size: %d \n",
    // conv_resource->bias_handle.GetBytesSize());

    if (conv_param->bias) {
        bias_term_ = true;
        if (output_dims[1] * sizeof(float) !=
            conv_resource->bias_handle.GetBytesSize()) {
            return TNNERR_MODEL_ERR;
        }

        const int bias_dim[] = {1, output_dims[1], 1, 1, 1};
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc_));
        CUDNN_CHECK(cudnnSetTensorNdDescriptorEx(
            bias_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 5, bias_dim));

        CUDA_CHECK(cudaMalloc((void **)&bias_,
                              conv_resource->bias_handle.GetBytesSize()));
        CUDA_CHECK(cudaMemcpy(
            bias_, conv_resource->bias_handle.force_to<float *>(),
            conv_resource->bias_handle.GetBytesSize(), cudaMemcpyHostToDevice));
    }

    return this->Reshape(inputs, outputs);
}

CudaConv3DLayerAcc::~CudaConv3DLayerAcc(){
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

Status CudaConv3DLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    DimsVector input_dims  = inputs[0]->GetBlobDesc().dims;
    DimsVector output_dims = outputs[0]->GetBlobDesc().dims;

    // LOGD("input n,c,d,h,w: %d, %d, %d, %d, %d , output n,c,h,w: %d, %d, %d,
    // %d, %d \n",
    //      blob_info_.batch, blob_info_.input_c, blob_info_.input_d,
    //      blob_info_.input_h, blob_info_.input_w,
    //      blob_info_.batch, blob_info_.output_c, blob_info_.output_d,
    //      blob_info_.output_h, blob_info_.output_w);

    int in_dims[] = {input_dims[0], input_dims[1], input_dims[2],
                     input_dims[3], input_dims[4]};
    CUDNN_CHECK(cudnnSetTensorNdDescriptorEx(bottom_desc_, CUDNN_TENSOR_NCHW,
                                             CUDNN_DATA_FLOAT, 5, in_dims));

    int out_dims[5];
    CUDNN_CHECK(cudnnGetConvolutionNdForwardOutputDim(
        conv_desc_, bottom_desc_, filter_desc_, 5, out_dims));

    // LOGD("conv3d layer acc cudnn infered ncdhw %d %d %d %d %d\n",
    // out_dims[0],
    //      out_dims[1], out_dims[2], out_dims[3], out_dims[4]);

    CUDNN_CHECK(cudnnSetTensorNdDescriptorEx(top_desc_, CUDNN_TENSOR_NCHW,
                                             CUDNN_DATA_FLOAT, 5, out_dims));

    for(int i=0;i<5;i++) {
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

    auto valid_algos = getValidAlgorithms(perf_results.get(), perf_count);
    if (valid_algos.size() == 0) {
        LOGE("CUDNN get conv algo failed.\n");
        return TNNERR_LAYER_ERR;
    }
    conv_algo_ = valid_algos[0].algo;

    // LOGD("Convolution algorithm: %d\n", conv_algo_);

    // workspace
    size_t needed_workspace_size;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        context_->cudnn_handle_, bottom_desc_, filter_desc_, conv_desc_,
        top_desc_, conv_algo_, &needed_workspace_size));

    // LOGD("Workspace size: %ld\n", workspace_size_);
    if (workspace_size_ < needed_workspace_size) {
        workspace_size_ = needed_workspace_size;
        if (workspace_data_ != nullptr) {
            CUDA_CHECK(cudaFree(workspace_data_));
        }
        CUDA_CHECK(cudaMalloc(&workspace_data_, workspace_size_));
    }
    return TNN_OK;
}

Status CudaConv3DLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
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

REGISTER_CUDA_ACC(Conv3D, LAYER_CONVOLUTION_3D);

}  // namespace TNN_NS
