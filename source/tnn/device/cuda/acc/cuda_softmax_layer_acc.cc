// Copyright 2019 Tencent. All Rights Reserved

#include "device/cuda/acc/cuda_softmax_layer_acc.h"
#include <iostream>
#include "device/cuda/cuda_utils.h"
#include "utils/dims_vector_utils.h"

namespace TNN_NS {

Status CudaSoftmaxLayerAcc::Init(Context *context, LayerParam *param,
                                 LayerResource *resource,
                                 const std::vector<Blob *> &inputs,
                                 const std::vector<Blob *> &outputs) {
    CudaLayerAcc::Init(context, param, resource, inputs, outputs);

    SoftmaxLayerParam *softmax_param = dynamic_cast<SoftmaxLayerParam *>(param);

    axis_ = softmax_param->axis;

    workspace_      = nullptr;
    workspace_size_ = 0;

    return this->Reshape(inputs, outputs);
}

CudaSoftmaxLayerAcc::~CudaSoftmaxLayerAcc() {
    if (workspace_ != nullptr) {
        CUDA_CHECK(cudaFree(workspace_));
    }
}

Status CudaSoftmaxLayerAcc::Reshape(const std::vector<Blob *> &inputs,
                                    const std::vector<Blob *> &outputs) {
    DimsVector input_dims  = inputs[0]->GetBlobDesc().dims;
    DimsVector output_dims = outputs[0]->GetBlobDesc().dims;

    FetchDimensions(inputs[0], outputs[0], blob_info_);

    outer_dim_ = DimsVectorUtils::Count(input_dims, 0, axis_);
    inner_dim_ = DimsVectorUtils::Count(input_dims, axis_ + 1);

    size_t required_size = inner_dim_ * outer_dim_ * sizeof(float);

    if (workspace_size_ < required_size) {
        workspace_size_ = required_size;
        if (workspace_ != nullptr) {
            CUDA_CHECK(cudaFree(workspace_));
        }
        CUDA_CHECK(cudaMalloc((void **)&workspace_, workspace_size_));
    }

    return TNN_OK;
}

CudaTypeLayerAccRegister<TypeLayerAccCreator<CudaSoftmaxLayerAcc>>
    g_cuda_softmax_layer_acc_register(LAYER_SOFTMAX);

}  // namespace TNN_NS
