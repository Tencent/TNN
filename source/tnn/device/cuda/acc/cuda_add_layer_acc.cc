// Copyright 2019 Tencent. All Rights Reserved

#include "device/cuda/acc/cuda_add_layer_acc.h"
#include <iostream>
#include "device/cuda/cuda_utils.h"
#include "utils/dims_vector_utils.h"

namespace TNN_NS {

Status CudaAddLayerAcc::Init(Context *context, LayerParam *param,
                             LayerResource *resource,
                             const std::vector<Blob *> &inputs,
                             const std::vector<Blob *> &outputs) {
    CudaLayerAcc::Init(context, param, resource, inputs, outputs);

    if (inputs.size() < 2) {
        return TNNERR_PARAM_ERR;
    }

    return this->Reshape(inputs, outputs);
}

CudaAddLayerAcc::~CudaAddLayerAcc() {}

Status CudaAddLayerAcc::Reshape(const std::vector<Blob *> &inputs,
                                const std::vector<Blob *> &outputs) {
    if (inputs.size() < 2) {
        return TNNERR_PARAM_ERR;
    }

    DimsVector input_dims  = inputs[0]->GetBlobDesc().dims;
    DimsVector output_dims = outputs[0]->GetBlobDesc().dims;

    FetchDimensions(inputs[0], outputs[0], blob_info_);

    return TNN_OK;
}

CudaTypeLayerAccRegister<TypeLayerAccCreator<CudaAddLayerAcc>>
    g_cuda_add_layer_acc_register(LAYER_ADD);

}  // namespace TNN_NS
