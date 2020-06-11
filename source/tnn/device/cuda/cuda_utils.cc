// Copyright 2019 Tencent. All Rights Reserved

#include "device/cuda/cuda_utils.h"
#include "core/blob.h"

namespace TNN_NS {

Status FetchDimensions3D(Blob *input, Blob *output, CudaLayerBlobInfo &info) {
    DimsVector input_dims  = input->GetBlobDesc().dims;
    DimsVector output_dims = output->GetBlobDesc().dims;

    info.batch = input_dims[0];

    info.input_c = input_dims[1];
    info.input_d = input_dims[2];
    info.input_h = input_dims[3];
    info.input_w = input_dims[4];

    info.output_c = output_dims[1];
    info.output_d = output_dims[2];
    info.output_h = output_dims[3];
    info.output_w = output_dims[4];

    // LOGD("FetchDimensions3D called\n");

    return TNN_OK;
}

Status FetchDimensions2D(Blob *input, Blob *output, CudaLayerBlobInfo &info) {
    DimsVector input_dims  = input->GetBlobDesc().dims;
    DimsVector output_dims = output->GetBlobDesc().dims;

    info.batch = input_dims[0];

    info.input_c = input_dims[1];
    info.input_h = input_dims[2];
    info.input_w = input_dims[3];
    info.input_d = 1;

    info.output_c = output_dims[1];
    info.output_h = output_dims[2];
    info.output_w = output_dims[3];
    info.output_d = 1;

    // LOGD("FetchDimensions2D called\n");

    return TNN_OK;
}

Status FetchDimensions(Blob *input, Blob *output, CudaLayerBlobInfo &info) {
    Status ret(TNN_OK);
    if (input->GetBlobDesc().data_format == DATA_FORMAT_NCDHW &&
        output->GetBlobDesc().data_format == DATA_FORMAT_NCDHW) {
        ret = FetchDimensions3D(input, output, info);
    } else {
        ret = FetchDimensions2D(input, output, info);
    }
    return ret;
}

Status FetchKernelInfo(ConvLayerParam *param, CudaLayerKernelInfo &info) {
    info.groups = param->group;

    info.kernel_h = param->kernels[1];
    info.kernel_w = param->kernels[0];
    if (param->kernels.size() > 2) {
        info.kernel_d = param->kernels[2];
    } else {
        info.kernel_d = 1;
    }

    info.pad_t = param->pads[2];  // H begin
    info.pad_b = param->pads[3];  // H end
    info.pad_l = param->pads[0];  // W begin
    info.pad_r = param->pads[1];  // W end
    if (param->pads.size() > 4) {
        info.pad_f = param->pads[4];  // D begin
        info.pad_e = param->pads[5];  // D end
    } else {
        info.pad_f = 0;
        info.pad_e = 0;
    }

    info.stride_h = param->strides[1];
    info.stride_w = param->strides[0];
    if (param->strides.size() > 2) {
        info.stride_d = param->strides[2];
    } else {
        info.stride_d = 1;
    }

    info.dilation_h = param->dialations[1];
    info.dilation_w = param->dialations[0];
    if (param->dialations.size() > 2) {
        info.dilation_d = param->dialations[2];
    } else {
        info.dilation_d = 1;
    }

    return TNN_OK;
}

Status FetchKernelInfo(PoolingLayerParam *param, CudaLayerKernelInfo &info) {
    info.kernel_h = param->kernels[1];
    info.kernel_w = param->kernels[0];
    if (param->kernels.size() > 2) {
        info.kernel_d = param->kernels[2];
    } else {
        info.kernel_d = 1;
    }

    info.pad_t = param->pads[2];  // H begin
    info.pad_b = param->pads[3];  // H end
    info.pad_l = param->pads[0];  // W begin
    info.pad_r = param->pads[1];  // W end
    if (param->pads.size() > 4) {
        info.pad_f = param->pads[4];  // D begin
        info.pad_e = param->pads[5];  // D end
    } else {
        info.pad_f = 0;
        info.pad_e = 0;
    }

    info.stride_h = param->strides[1];
    info.stride_w = param->strides[0];
    if (param->strides.size() > 2) {
        info.stride_d = param->strides[2];
    } else {
        info.stride_d = 1;
    }

    return TNN_OK;
}

Status DimsVectorTo4DArray(int *shape, DimsVector dims) {
    shape[0] = dims[0];
    shape[1] = dims[1];
    shape[2] = dims[2];
    int mul  = 1;
    for (int i = 3; i < dims.size(); i++) {
        mul *= dims[i];
    }
    shape[3] = mul;

    return TNN_OK;
}

}  // namespace TNN_NS
