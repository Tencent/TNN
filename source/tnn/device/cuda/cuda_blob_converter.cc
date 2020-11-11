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
#include "tnn/device/cuda/utils/cuda_blob_converter_kernel.cuh"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

CudaBlobConverterAcc::CudaBlobConverterAcc(Blob *blob) : BlobConverterAcc(blob) {
}

CudaBlobConverterAcc::~CudaBlobConverterAcc() {
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

    if (image.GetMatType() == NCHW_FLOAT) {
        cudaError_t status = cudaMemcpyAsync(image.GetData(), blob_data, DimsVectorUtils::Count(dims) * sizeof(float),
            cudaMemcpyDeviceToHost, stream);
        if (cudaSuccess != status) {
            return TNNERR_CUDA_MEMCPY_ERROR;
        }
    } else if (image.GetMatType() == N8UC4) {
        BlobToBGR(dims[0], chw, hw, blob_data, (unsigned char *)image.GetData(), stream, 4, param.scale.data(), param.bias.data());
    } else if (image.GetMatType() == N8UC3) {
        BlobToBGR(dims[0], chw, hw, blob_data, (unsigned char *)image.GetData(), stream, 3, param.scale.data(), param.bias.data());
    } else if (image.GetMatType() == NGRAY) {
        BlobToGray(nchw, blob_data, (unsigned char *)image.GetData(), stream, param.scale[0], param.bias[0]);
    } else {
        ret = Status(TNNERR_PARAM_ERR, "convert type not support yet");
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

    if (image.GetMatType() == NCHW_FLOAT) {
        cudaError_t status = cudaMemcpyAsync(blob_data, image.GetData(), DimsVectorUtils::Count(dims) * sizeof(float),
            cudaMemcpyHostToDevice, stream);
        if (cudaSuccess != status) {
            return TNNERR_CUDA_MEMCPY_ERROR;
        }
    } else if (image.GetMatType() == N8UC4) {
        BGRToBlob(dims[0], chw, hw, (unsigned char *)image.GetData(), blob_data, stream, 4, param.scale.data(), param.bias.data());
    } else if (image.GetMatType() == N8UC3) {
        BGRToBlob(dims[0], chw, hw, (unsigned char *)image.GetData(), blob_data, stream, 3, param.scale.data(), param.bias.data());
    } else if (image.GetMatType() == NGRAY) {
        GrayToBlob(nchw, (unsigned char *)image.GetData(), blob_data, stream, param.scale[0], param.bias[0]);
    } else {
        ret = Status(TNNERR_PARAM_ERR, "convert type not support yet");
    }
    return ret;
}

DECLARE_BLOB_CONVERTER_CREATER(Cuda);
REGISTER_BLOB_CONVERTER(Cuda, DEVICE_CUDA);

}  //  namespace TNN_NS