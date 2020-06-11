// Copyright 2019 Tencent. All Rights Reserved

#include "device/cuda/acc/cuda_reduce_mean_layer_acc.h"
#include <iostream>
#include "device/cuda/cuda_utils.h"
#include "utils/dims_vector_utils.h"

namespace TNN_NS {

Status CudaReduceMeanLayerAcc::Init(Context *context, LayerParam *param,
                                    LayerResource *resource,
                                    const std::vector<Blob *> &inputs,
                                    const std::vector<Blob *> &outputs) {
    CudaLayerAcc::Init(context, param, resource, inputs, outputs);

    ReduceMeanLayerParam *reduce_mean_param =
        dynamic_cast<ReduceMeanLayerParam *>(param);
    axis_ = reduce_mean_param->axis[0];

    return this->Reshape(inputs, outputs);
}

CudaReduceMeanLayerAcc::~CudaReduceMeanLayerAcc() {}

Status CudaReduceMeanLayerAcc::Reshape(const std::vector<Blob *> &inputs,
                                       const std::vector<Blob *> &outputs) {
    DimsVector input_dims  = inputs[0]->GetBlobDesc().dims;
    DimsVector output_dims = outputs[0]->GetBlobDesc().dims;

    FetchDimensions(inputs[0], outputs[0], blob_info_);

    outer_dim_ = DimsVectorUtils::Count(input_dims, 0, axis_);
    inner_dim_ = DimsVectorUtils::Count(input_dims, axis_ + 1);

    return TNN_OK;
}

CudaTypeLayerAccRegister<TypeLayerAccCreator<CudaReduceMeanLayerAcc>>
    g_cuda_reduce_mean_layer_acc_register(LAYER_REDUCE_MEAN);

}  // namespace TNN_NS
