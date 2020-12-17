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
    scale_ptr_ = nullptr;
    bias_ptr_ = nullptr;
    image_ptr_ = nullptr;
}

CudaBlobConverterAcc::~CudaBlobConverterAcc() {
    if (scale_ptr_) cudaFree(scale_ptr_);
    if (bias_ptr_) cudaFree(bias_ptr_);
    if (image_ptr_) cudaFree(image_ptr_);
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
    auto hw = DimsVectorUtils::Count(dims, 2);
    auto chw = dims[1] * hw;
    auto nchw = dims[0] * chw;

    if (image.GetMatType() == NCHW_FLOAT) {
        if (!scale_ptr_) cudaMalloc((void**)&scale_ptr_, dims[1] * sizeof(float));
        if (!bias_ptr_) cudaMalloc((void**)&bias_ptr_, dims[1] * sizeof(float));
        if (!image_ptr_ || DimsVectorUtils::Count(dims) * sizeof(float) > image_size_) {
            if (image_ptr_) cudaFree(image_ptr_);
            cudaMalloc((void**)&image_ptr_, DimsVectorUtils::Count(dims) * sizeof(float));
            image_size_ = DimsVectorUtils::Count(dims) * sizeof(float);
        }
        if (param.scale.size() == dims[1]) {
            cudaMemcpy(scale_ptr_, param.scale.data(), dims[1] * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(bias_ptr_, param.bias.data(), dims[1] * sizeof(float), cudaMemcpyHostToDevice);
            ScaleBias(blob_data, (float*)image_ptr_, stream, scale_ptr_, bias_ptr_, dims[0], dims[1], hw);
            cudaMemcpy(image.GetData(), image_ptr_, DimsVectorUtils::Count(dims) * sizeof(float), cudaMemcpyDeviceToHost);
        } else {
            cudaMemcpy(image.GetData(), blob_data, DimsVectorUtils::Count(dims) * sizeof(float), cudaMemcpyDeviceToHost);
        }
    } else if (image.GetMatType() == N8UC4) {
        if (!scale_ptr_) cudaMalloc((void**)&scale_ptr_, 4 * sizeof(float));
        if (!bias_ptr_) cudaMalloc((void**)&bias_ptr_, 4 * sizeof(float));
        if (!image_ptr_) {
            cudaMalloc((void**)&image_ptr_, DimsVectorUtils::Count(dims) * sizeof(unsigned char));
            image_size_ = dims[0] * 4 * hw * sizeof(unsigned char);
        }
        cudaMemcpy(scale_ptr_, param.scale.data(), 4 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(bias_ptr_, param.bias.data(), 4 * sizeof(float), cudaMemcpyHostToDevice);
        BlobToBGR(dims[0], chw, hw, blob_data, (unsigned char*)image_ptr_, stream, 4, scale_ptr_, bias_ptr_, param.reverse_channel);
        cudaMemcpy(image.GetData(), image_ptr_, dims[0] * 4 * hw * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    } else if (image.GetMatType() == N8UC3) {
        if (!scale_ptr_) cudaMalloc((void**)&scale_ptr_, 4 * sizeof(float));
        if (!bias_ptr_) cudaMalloc((void**)&bias_ptr_, 4 * sizeof(float));
        if (!image_ptr_) {
            cudaMalloc((void**)&image_ptr_, DimsVectorUtils::Count(dims) * sizeof(unsigned char));
            image_size_ = DimsVectorUtils::Count(dims) * sizeof(unsigned char);
        }
        cudaMemcpy(scale_ptr_, param.scale.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(bias_ptr_, param.bias.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);
        BlobToBGR(dims[0], chw, hw, blob_data, (unsigned char*)image_ptr_, stream, 3, scale_ptr_, bias_ptr_, param.reverse_channel);
        cudaMemcpy(image.GetData(), image_ptr_, DimsVectorUtils::Count(dims) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    } else if (image.GetMatType() == NGRAY) {
        if (!image_ptr_) {
            cudaMalloc((void**)&image_ptr_, DimsVectorUtils::Count(dims) * sizeof(unsigned char));
            image_size_ = DimsVectorUtils::Count(dims) * sizeof(unsigned char);
        }
        BlobToGray(nchw, blob_data, (unsigned char*)image_ptr_, stream, param.scale[0], param.bias[0]);
        cudaMemcpy(image.GetData(), image_ptr_, DimsVectorUtils::Count(dims) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    } else if (desc.data_type == DATA_TYPE_INT32) {
        if (image.GetMatType() == NC_INT32) {
            cudaMemcpy(image.GetData(), blob_data, DimsVectorUtils::Count(dims) * sizeof(int32_t), cudaMemcpyDeviceToHost);
        } else {
            ret = Status(TNNERR_PARAM_ERR, "blob.data_type and mat.type not match");
        }
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
        if (!scale_ptr_) cudaMalloc((void**)&scale_ptr_, dims[1] * sizeof(float));
        if (!bias_ptr_) cudaMalloc((void**)&bias_ptr_, dims[1] * sizeof(float));
        if (!image_ptr_ || DimsVectorUtils::Count(dims) * sizeof(float) > image_size_) {
            if (image_ptr_) cudaFree(image_ptr_);
            cudaMalloc((void**)&image_ptr_, DimsVectorUtils::Count(dims) * sizeof(float));
            image_size_ = DimsVectorUtils::Count(dims) * sizeof(float);
        }
        cudaMemcpy(scale_ptr_, param.scale.data(), dims[1] * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(bias_ptr_, param.bias.data(), dims[1] * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(image_ptr_, image.GetData(), DimsVectorUtils::Count(dims) * sizeof(float), cudaMemcpyHostToDevice);
        ScaleBias((float*)image_ptr_, blob_data, stream, scale_ptr_, bias_ptr_, dims[0], dims[1], hw);
    } else if (image.GetMatType() == N8UC4) {
        if (!scale_ptr_) cudaMalloc((void**)&scale_ptr_, 4 * sizeof(float));
        if (!bias_ptr_) cudaMalloc((void**)&bias_ptr_, 4 * sizeof(float));
        if (!image_ptr_) {
            cudaMalloc((void**)&image_ptr_, dims[0] * 4 * hw * sizeof(unsigned char));
            image_size_ = dims[0] * 4 * hw * sizeof(unsigned char);
        }
        cudaMemcpy(image_ptr_, image.GetData(), dims[0] * 4 * hw * sizeof(unsigned char), cudaMemcpyHostToDevice);
        cudaMemcpy(scale_ptr_, param.scale.data(), 4 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(bias_ptr_, param.bias.data(), 4 * sizeof(float), cudaMemcpyHostToDevice);
        BGRToBlob(dims[0], chw, hw, (unsigned char*)image_ptr_, blob_data, stream, 4, scale_ptr_, bias_ptr_, param.reverse_channel);
    } else if (image.GetMatType() == N8UC3) {
        if (!scale_ptr_) cudaMalloc((void**)&scale_ptr_, 4 * sizeof(float));
        if (!bias_ptr_) cudaMalloc((void**)&bias_ptr_, 4 * sizeof(float));
        if (!image_ptr_) {
            cudaMalloc((void**)&image_ptr_, DimsVectorUtils::Count(dims) * sizeof(unsigned char));
            image_size_ = DimsVectorUtils::Count(dims) * sizeof(unsigned char);
        }
        cudaMemcpy(image_ptr_, image.GetData(), DimsVectorUtils::Count(dims) * sizeof(unsigned char), cudaMemcpyHostToDevice);
        cudaMemcpy(scale_ptr_, param.scale.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(bias_ptr_, param.bias.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);
        BGRToBlob(dims[0], chw, hw, (unsigned char*)image_ptr_, blob_data, stream, 3, scale_ptr_, bias_ptr_, param.reverse_channel);
    } else if (image.GetMatType() == NGRAY) {
        if (!image_ptr_) {
            cudaMalloc((void**)&image_ptr_, DimsVectorUtils::Count(dims) * sizeof(unsigned char));
            image_size_ = DimsVectorUtils::Count(dims) * sizeof(unsigned char);
        }
        cudaMemcpy(image_ptr_, image.GetData(), DimsVectorUtils::Count(dims) * sizeof(unsigned char), cudaMemcpyHostToDevice);
        GrayToBlob(nchw, (unsigned char*)image_ptr_, blob_data, stream, param.scale[0], param.bias[0]);
    } else if (image.GetMatType() == NC_INT32 ) {
        desc.data_type = DATA_TYPE_INT32;
        blob_->SetBlobDesc(desc);
        cudaMemcpy(blob_data, image.GetData(), DimsVectorUtils::Count(dims) * sizeof(int32_t), cudaMemcpyHostToDevice);
    } else {
        ret = Status(TNNERR_PARAM_ERR, "convert type not support yet");
    }
    return ret;
}

DECLARE_BLOB_CONVERTER_CREATER(Cuda);
REGISTER_BLOB_CONVERTER(Cuda, DEVICE_CUDA);

}  //  namespace TNN_NS

