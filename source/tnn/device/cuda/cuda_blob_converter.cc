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

#include "tnn/device/cuda/cuda_blob_converter.h"
#include "tnn/device/cuda/cuda_context.h"
#include "tnn/device/cuda/cuda_device.h"
#include "tnn/device/cuda/cuda_blob_converter_kernel.cuh"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

CudaBlobConverterAcc::CudaBlobConverterAcc(Blob *blob) : BlobConverterAcc(blob) {
    scale_ptr = nullptr;
    bias_ptr = nullptr;
    image_ptr = nullptr;
}

CudaBlobConverterAcc::~CudaBlobConverterAcc() {
    if (scale_ptr) cudaFree(scale_ptr);
    if (bias_ptr) cudaFree(bias_ptr);
    if (image_ptr) cudaFree(image_ptr);
}

Status CudaBlobConverterAcc::ConvertToMat(Mat& image, MatConvertParam param, void* command_queue) {
    Status ret = ConvertToMatAsync(image, param, command_queue);
    if (ret != TNN_OK) {
        return ret;
    }
    cudaStream_t stream = static_cast<cudaStream_t>(command_queue);
    cudaError_t status = cudaStreamSynchronize(stream);
    if (cudaSuccess != status) {
        return TNNERR_CUDA_SYNC_ERROR;
    }
    return TNN_OK;
}

Status CudaBlobConverterAcc::ConvertToMatAsync(Mat& image, MatConvertParam param, void* command_queue) {
    Status ret = TNN_OK;
    if (blob_ == nullptr) {
        return Status(TNNERR_NULL_PARAM, "input/output blob is null");
    }

    cudaStream_t stream = static_cast<cudaStream_t>(command_queue);
    auto blob_data = reinterpret_cast<float*>(blob_->GetHandle().base);
    auto desc = blob_->GetBlobDesc();
    auto dims = desc.dims;
    auto hw = dims[2] * dims[3];
    auto chw = dims[1] * hw;
    auto nchw = dims[0] * chw;

    if (image.GetDeviceType() == DEVICE_CUDA) {
        prepareParamPtr(param, image.GetMatType(), stream);
        if (image.GetMatType() == NCHW_FLOAT) {
            ScaleBias(blob_data, (float*)image.GetData(), stream, scale_ptr, bias_ptr, dims[0], dims[1], hw);
        } else if (image.GetMatType() == N8UC4) {
            BlobToBGR(dims[0], chw, hw, blob_data, (unsigned char*)image.GetData(), stream, 4, scale_ptr, bias_ptr,
                param.reverse_channel);
        } else if (image.GetMatType() == N8UC3) {
            BlobToBGR(dims[0], chw, hw, blob_data, (unsigned char*)image.GetData(), stream, 3, scale_ptr, bias_ptr,
                param.reverse_channel);
        } else if (image.GetMatType() == NGRAY) {
            BlobToGray(nchw, blob_data, (unsigned char*)image.GetData(), stream, param.scale[0], param.bias[0]);
        } else {
            ret = Status(TNNERR_PARAM_ERR, "convert type not support yet");
        }
    } else {
        prepareImagePtr(image, param, dims, stream);
        prepareParamPtr(param, image.GetMatType(), stream);
        if (image.GetMatType() == NCHW_FLOAT) {
            ScaleBias(blob_data, (float*)image_ptr, stream, scale_ptr, bias_ptr, dims[0], dims[1], hw);
            cudaMemcpyAsync(image.GetData(), image_ptr, DimsVectorUtils::Count(dims) * sizeof(float),
                cudaMemcpyDeviceToHost, stream);
        } else if (image.GetMatType() == N8UC4) {
            BlobToBGR(dims[0], chw, hw, blob_data, (unsigned char*)image_ptr, stream, 4, scale_ptr, bias_ptr,
                param.reverse_channel);
            cudaMemcpyAsync(image.GetData(), image_ptr, dims[0] * 4 * hw * sizeof(unsigned char),
                cudaMemcpyDeviceToHost, stream);
        } else if (image.GetMatType() == N8UC3) {
            BlobToBGR(dims[0], chw, hw, blob_data, (unsigned char*)image_ptr, stream, 3, scale_ptr, bias_ptr,
                param.reverse_channel);
            cudaMemcpyAsync(image.GetData(), image_ptr, DimsVectorUtils::Count(dims) * sizeof(unsigned char),
                cudaMemcpyDeviceToHost, stream);
        } else if (image.GetMatType() == NGRAY) {
            BlobToGray(nchw, blob_data, (unsigned char*)image_ptr, stream, param.scale[0], param.bias[0]);
            cudaMemcpyAsync(image.GetData(), image_ptr, DimsVectorUtils::Count(dims) * sizeof(unsigned char),
                cudaMemcpyDeviceToHost, stream);
        } else {
            ret = Status(TNNERR_PARAM_ERR, "convert type not support yet");
        }
    }
    return ret;
}

Status CudaBlobConverterAcc::ConvertFromMat(Mat& image, MatConvertParam param, void* command_queue) {
    Status ret = ConvertFromMatAsync(image, param, command_queue);
    if (ret != TNN_OK) {
        return ret;
    }
    cudaStream_t stream = static_cast<cudaStream_t>(command_queue);
    cudaError_t status = cudaStreamSynchronize(stream);
    if (cudaSuccess != status) {
        return TNNERR_CUDA_SYNC_ERROR;
    }
    return TNN_OK;
}

Status CudaBlobConverterAcc::ConvertFromMatAsync(Mat& image, MatConvertParam param, void* command_queue) {
    Status ret = TNN_OK;
    if (blob_ == nullptr) {
        return Status(TNNERR_NULL_PARAM, "input/output blob_ is null");
    }

    cudaStream_t stream = static_cast<cudaStream_t>(command_queue);
    auto blob_data = reinterpret_cast<float*>(blob_->GetHandle().base);
    auto desc = blob_->GetBlobDesc();
    auto dims = desc.dims;
    auto hw = dims[2] * dims[3];
    auto chw = dims[1] * hw;
    auto nchw = dims[0] * chw;

    if (image.GetDeviceType() == DEVICE_CUDA) {
        prepareParamPtr(param, image.GetMatType(), command_queue);
        if (image.GetMatType() == NCHW_FLOAT) {
            ScaleBias(blob_data, (float*)image.GetData(), stream, scale_ptr, bias_ptr, dims[0], dims[1], hw);
        } else if (image.GetMatType() == N8UC4) {
            BGRToBlob(dims[0], chw, hw, (unsigned char*)image.GetData(), blob_data, stream, 4, scale_ptr, bias_ptr,
                param.reverse_channel);
        } else if (image.GetMatType() == N8UC3) {
            BGRToBlob(dims[0], chw, hw, (unsigned char*)image.GetData(), blob_data, stream, 3, scale_ptr, bias_ptr,
                param.reverse_channel);
        } else if (image.GetMatType() == NGRAY) {
            GrayToBlob(nchw, (unsigned char*)image.GetData(), blob_data, stream, param.scale[0], param.bias[0]);
        } else {
            ret = Status(TNNERR_PARAM_ERR, "convert type not support yet");
        }
    } else {
        prepareImagePtr(image, param, dims, command_queue);
        prepareParamPtr(param, image.GetMatType(), command_queue);
        if (image.GetMatType() == NCHW_FLOAT) {
            cudaMemcpyAsync(image_ptr, image.GetData(), DimsVectorUtils::Count(dims) * sizeof(float),
                cudaMemcpyHostToDevice, stream);
            ScaleBias((float*)image_ptr, blob_data, stream, scale_ptr, bias_ptr, dims[0], dims[1], hw);
        } else if (image.GetMatType() == N8UC4) {
            cudaMemcpyAsync(image_ptr, image.GetData(), dims[0] * 4 * hw * sizeof(unsigned char),
                cudaMemcpyHostToDevice, stream);
            BGRToBlob(dims[0], chw, hw, (unsigned char*)image_ptr, blob_data, stream, 4, scale_ptr, bias_ptr,
                param.reverse_channel);
        } else if (image.GetMatType() == N8UC3) {
            cudaMemcpyAsync(image_ptr, image.GetData(), DimsVectorUtils::Count(dims) * sizeof(unsigned char),
                cudaMemcpyHostToDevice, stream);
            BGRToBlob(dims[0], chw, hw, (unsigned char*)image_ptr, blob_data, stream, 3, scale_ptr, bias_ptr,
                param.reverse_channel);
        } else if (image.GetMatType() == NGRAY) {
            cudaMemcpyAsync(image_ptr, image.GetData(), DimsVectorUtils::Count(dims) * sizeof(unsigned char),
                cudaMemcpyHostToDevice, stream);
            GrayToBlob(nchw, (unsigned char*)image_ptr, blob_data, stream, param.scale[0], param.bias[0]);
        } else {
            ret = Status(TNNERR_PARAM_ERR, "convert type not support yet");
        }
    }

    return ret;
}

void CudaBlobConverterAcc::prepareImagePtr(Mat& image, MatConvertParam param, DimsVector dims, void* command_queue) {
    cudaStream_t stream = static_cast<cudaStream_t>(command_queue);
    int hw = dims[2] * dims[3];
    int n = dims[0];
    int unitBytes = image.GetMatType() == NCHW_FLOAT ? sizeof(float) : sizeof(unsigned char);
    int current_image_size = image.GetMatType() == N8UC4 ? hw * 4 * n : DimsVectorUtils::Count(dims);
    current_image_size *= unitBytes;
    if (!image_ptr || current_image_size > image_size) {
        if (image_ptr) cudaFree(image_ptr);
        cudaMalloc((void**)&image_ptr, current_image_size);
        image_size = current_image_size;
    }
}

void CudaBlobConverterAcc::prepareParamPtr(MatConvertParam param, MatType type, void* command_queue) {
    if (type == NGRAY) return;
    cudaStream_t stream = static_cast<cudaStream_t>(command_queue);
    int c_reserve;
    int c_copy;
    if (type == N8UC4) {
        c_reserve = 4;
        c_copy = 4;
    } else if (type == NCHW_FLOAT) {
        c_reserve = blob_->GetBlobDesc().dims[1];
        c_copy = c_reserve;
    } else {
        c_reserve = 4;
        c_copy = 3;
    }
    if (!scale_ptr) cudaMalloc((void**)&scale_ptr, c_reserve * sizeof(float));
    if (!bias_ptr) cudaMalloc((void**)&bias_ptr, c_reserve * sizeof(float));
    cudaMemcpyAsync(scale_ptr, param.scale.data(), c_copy * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(bias_ptr, param.bias.data(), c_copy * sizeof(float), cudaMemcpyHostToDevice, stream);
}

DECLARE_BLOB_CONVERTER_CREATER(Cuda);
REGISTER_BLOB_CONVERTER(Cuda, DEVICE_CUDA);

}  //  namespace TNN_NS


