// Copyright 2019 Tencent. All Rights Reserved

#include "device/cuda/acc/cuda_pooling_layer_acc.h"
#include <iostream>
#include "device/cuda/cuda_utils.h"
#include "utils/dims_vector_utils.h"

namespace TNN_NS {

Status CudaPoolingLayerAcc::Init(Context *context, LayerParam *param,
                                 LayerResource *resource,
                                 const std::vector<Blob *> &inputs,
                                 const std::vector<Blob *> &outputs) {
    CudaLayerAcc::Init(context, param, resource, inputs, outputs);
    alpha_ = 1.0f;
    beta_  = 0.0f;

    PoolingLayerParam *pooling_param = dynamic_cast<PoolingLayerParam *>(param);
    FetchKernelInfo(pooling_param, kernel_);

    // LOGD("pooling pad size:%lu value:%d %d %d\n",
    //     pooling_param->pads.size(),
    //     pooling_param->pads[0], pooling_param->pads[1],
    //     pooling_param->pads[2]);

    if (pooling_param->pool_type == 0) {
        pooling_mode_ = CUDNN_POOLING_MAX;
    } else if (pooling_param->pool_type == 1) {
        pooling_mode_ = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
    } else {
        pooling_mode_ = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    }

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bottom_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&top_desc_));
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_desc_));

    LOGD("pooling kernel whd:%d %d %d\n", kernel_.kernel_w, kernel_.kernel_h,
         kernel_.kernel_d);

    return this->Reshape(inputs, outputs);
}

CudaPoolingLayerAcc::~CudaPoolingLayerAcc() {
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bottom_desc_));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_desc_));
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pool_desc_));
}

Status CudaPoolingLayerAcc::Reshape(const std::vector<Blob *> &inputs,
                                    const std::vector<Blob *> &outputs) {
    DimsVector input_dims  = inputs[0]->GetBlobDesc().dims;
    DimsVector output_dims = outputs[0]->GetBlobDesc().dims;

    Blob *input  = inputs[0];
    Blob *output = inputs[0];

    FetchDimensions(input, output, blob_info_);

    int in_dims[] = {blob_info_.batch, blob_info_.input_c, blob_info_.input_d,
                     blob_info_.input_h, blob_info_.input_w};
    CUDNN_CHECK(cudnnSetTensorNdDescriptorEx(bottom_desc_, CUDNN_TENSOR_NCHW,
                                             CUDNN_DATA_FLOAT, 5, in_dims));

    const int pad_dims[] = {kernel_.pad_f, kernel_.pad_t,
                            kernel_.pad_l};  // DHW
    const int sti_dims[] = {kernel_.stride_d, kernel_.stride_h,
                            kernel_.stride_w};
    const int kel_dims[] = {kernel_.kernel_d, kernel_.kernel_h,
                            kernel_.kernel_w};
    LOGD("pad dhw %d %d %d, str:%d %d %d, ker:%d %d %d\n", pad_dims[0],
         pad_dims[1], pad_dims[2], sti_dims[0], sti_dims[1], sti_dims[2],
         kel_dims[0], kel_dims[1], kel_dims[2]);

    CUDNN_CHECK(cudnnSetPoolingNdDescriptor(pool_desc_, pooling_mode_,
                                            CUDNN_PROPAGATE_NAN, 3, kel_dims,
                                            pad_dims, sti_dims));

    int out_dims[5];
    CUDNN_CHECK(cudnnGetPoolingNdForwardOutputDim(pool_desc_, bottom_desc_, 5,
                                                  out_dims));

    LOGD("pooling layer acc cudnn infered ncdhw %d %d %d %d %d\n", out_dims[0],
         out_dims[1], out_dims[2], out_dims[3], out_dims[4]);

    CUDNN_CHECK(cudnnSetTensorNdDescriptorEx(top_desc_, CUDNN_TENSOR_NCHW,
                                             CUDNN_DATA_FLOAT, 5, out_dims));

    return TNN_OK;
}

Status CudaPoolingLayerAcc::Forward(const std::vector<Blob *> &inputs,
                                    const std::vector<Blob *> &outputs) {
    CUDNN_CHECK(cudnnPoolingForward(context_->cudnn_handle_, pool_desc_,
                                    &alpha_, bottom_desc_,
                                    inputs[0]->GetHandle().base, &beta_,
                                    top_desc_, outputs[0]->GetHandle().base));

    CUDA_CHECK(cudaStreamSynchronize(context_->stream_));

    return TNN_OK;
}

CudaTypeLayerAccRegister<TypeLayerAccCreator<CudaPoolingLayerAcc>>
    g_cuda_pooling_layer_acc_register(LAYER_POOLING);

CudaTypeLayerAccRegister<TypeLayerAccCreator<CudaPoolingLayerAcc>>
    g_cuda_pooling_3d_layer_acc_register(LAYER_POOLING_3D);

}  // namespace TNN_NS
