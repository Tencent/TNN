// Copyright 2019 Tencent. All Rights Reserved

#include "device/cuda/acc/cuda_reshape_layer_acc.h"
#include <iostream>
#include "device/cuda/cuda_utils.h"
#include "utils/dims_vector_utils.h"

namespace TNN_NS {

Status CudaReshapeLayerAcc::Init(Context *context, LayerParam *param,
                                 LayerResource *resource,
                                 const std::vector<Blob *> &inputs,
                                 const std::vector<Blob *> &outputs) {
    CudaLayerAcc::Init(context, param, resource, inputs, outputs);

    return this->Reshape(inputs, outputs);
}

CudaReshapeLayerAcc::~CudaReshapeLayerAcc() {}

Status CudaReshapeLayerAcc::Reshape(const std::vector<Blob *> &inputs,
                                    const std::vector<Blob *> &outputs) {
    DimsVector input_dims  = inputs[0]->GetBlobDesc().dims;
    DimsVector output_dims = outputs[0]->GetBlobDesc().dims;

    FetchDimensions(inputs[0], outputs[0], blob_info_);

    return TNN_OK;
}

Status CudaReshapeLayerAcc::Forward(const std::vector<Blob *> &inputs,
                                    const std::vector<Blob *> &outputs) {
    float *bottom_data = (float *)inputs[0]->GetHandle().base;
    float *top_data    = (float *)outputs[0]->GetHandle().base;
    size_t size =
        DimsVectorUtils::Count(inputs[0]->GetBlobDesc().dims) * sizeof(float);

    CUDA_CHECK(cudaMemcpyAsync(top_data, bottom_data, size,
                               cudaMemcpyDeviceToDevice, context_->stream_));

    return TNN_OK;
}

CudaTypeLayerAccRegister<TypeLayerAccCreator<CudaReshapeLayerAcc>>
    g_cuda_reshape_layer_acc_register(LAYER_RESHAPE);

}  // namespace TNN_NS
