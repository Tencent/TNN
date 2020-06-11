// Copyright 2019 Tencent. All Rights Reserved

#include "device/cuda/acc/cuda_conv_layer_acc.h"
#include "device/cuda/cuda_utils.h"

namespace TNN_NS {

Status CudaConvLayerAcc::Init(Context *context, LayerParam *param,
                              LayerResource *resource,
                              const std::vector<Blob *> &inputs,
                              const std::vector<Blob *> &outputs) {
    CudaLayerAcc::Init(context, param, resource, inputs, outputs);
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param);
    LOGD("CudaConvLayer param kernel_w: %d, kernel_h: %d \n",
         conv_param->kernels[0], conv_param->kernels[1]);

    ConvLayerResource *conv_resource =
        dynamic_cast<ConvLayerResource *>(resource);
    float *weights = conv_resource->filter_handle.force_to<float *>();
    LOGD("weights0: %f \n", weights[0]);

    DimsVector input_dims  = inputs[0]->GetBlobDesc().dims;
    DimsVector output_dims = outputs[0]->GetBlobDesc().dims;
    LOGD("input n,c,h,w: %d, %d, %d, %d , output n,c,h,w: %d, %d, %d, %d \n",
         input_dims[0], input_dims[1], input_dims[2], input_dims[3],
         output_dims[0], output_dims[1], output_dims[2], output_dims[3]);

    FetchDimensions(inputs[0], outputs[0], blob_info_);

    return TNN_OK;
}

CudaConvLayerAcc::~CudaConvLayerAcc() {}

Status CudaConvLayerAcc::Reshape(const std::vector<Blob *> &inputs,
                                 const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaConvLayerAcc::Forward(const std::vector<Blob *> &inputs,
                                 const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

// CudaTypeLayerAccRegister<TypeLayerAccCreator<CudaConvLayerAcc>>
//     g_cuda_conv_layer_acc_register(LAYER_CONVOLUTION);

}  // namespace TNN_NS
