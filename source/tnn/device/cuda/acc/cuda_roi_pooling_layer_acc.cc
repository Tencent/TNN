// Copyright 2019 Tencent. All Rights Reserved

#include "device/cuda/acc/cuda_roi_pooling_layer_acc.h"
#include <iostream>
#include "device/cuda/cuda_utils.h"
#include "utils/dims_vector_utils.h"

namespace TNN_NS {

Status CudaRoiPoolingLayerAcc::Init(Context *context, LayerParam *param,
                                    LayerResource *resource,
                                    const std::vector<Blob *> &inputs,
                                    const std::vector<Blob *> &outputs) {
    CudaLayerAcc::Init(context, param, resource, inputs, outputs);

    RoiPoolingLayerParam *pooling_param =
        dynamic_cast<RoiPoolingLayerParam *>(param);

    if (pooling_param->pooled_dims.size() == 3) {
        LOGD("pooled kernel whd:%d %d %d\n", pooling_param->pooled_dims[0],
             pooling_param->pooled_dims[1], pooling_param->pooled_dims[2]);
    }

    return this->Reshape(inputs, outputs);
}

CudaRoiPoolingLayerAcc::~CudaRoiPoolingLayerAcc() {}

Status CudaRoiPoolingLayerAcc::Reshape(const std::vector<Blob *> &inputs,
                                       const std::vector<Blob *> &outputs) {
    DimsVector input_dims  = inputs[0]->GetBlobDesc().dims;
    DimsVector output_dims = outputs[0]->GetBlobDesc().dims;

    Blob *input  = inputs[0];
    Blob *output = inputs[0];

    FetchDimensions(input, output, blob_info_);

    return TNN_OK;
}

CudaTypeLayerAccRegister<TypeLayerAccCreator<CudaRoiPoolingLayerAcc>>
    g_cuda_roi_pooling_layer_acc_register(LAYER_ROIPOOLING);

}  // namespace TNN_NS
