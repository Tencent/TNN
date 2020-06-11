// Copyright 2019 Tencent. All Rights Reserved

#include "device/cuda/acc/cuda_layer_acc.h"
#include "device/cuda/cuda_utils.h"
#include "utils/dims_vector_utils.h"

namespace TNN_NS {

CudaLayerAcc::~CudaLayerAcc() {
    return;
}

Status CudaLayerAcc::Init(Context *context, LayerParam *param,
                          LayerResource *resource,
                          const std::vector<Blob *> &inputs,
                          const std::vector<Blob *> &outputs) {
    AbstractLayerAcc::Init(context, param, resource, inputs, outputs);
    context_ = dynamic_cast<CudaContext *>(context);
    param_   = param;

    return TNN_OK;
}

std::vector<DataFormat> CudaLayerAcc::SupportDataFormat(DataType data_type,
                                                        int dims_size) {
    std::vector<DataFormat> support_list;
    if (dims_size == 4) {
        support_list.push_back(DATA_FORMAT_NC4HW4);
    } else if (dims_size == 5) {
        support_list.push_back(DATA_FORMAT_NCDHW);
    }
    return support_list;
}

}  // namespace TNN_NS
