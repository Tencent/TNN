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

#include <cuda_runtime.h>

#include "tnn/core/blob_int8.h"
#include "tnn/core/macro.h"
#include "tnn/utils/naive_compute.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/bfp16_utils.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/mat_converter_utils.h"
#include "tnn/device/cuda/cuda_mat_converter.h"
#include "tnn/device/cuda/cuda_mat_util.cuh"

namespace TNN_NS {

Status CudaMatConverterAcc::Copy(Mat& src, Mat& dst, void* command_queue) {
    Status ret = TNN_OK;

    ret = CheckMatConverterParams(src, dst, false);
    if (ret != TNN_OK)
        return ret;

    MatType mat_type = src.GetMatType();
    int data_type_size = 1;
    DimsVector dims    = src.GetDims();
    if (mat_type == NCHW_FLOAT) {
        data_type_size = sizeof(float);
    } else if (mat_type == N8UC4) {
        //special for 8UC4, blob channel <= 4.
        dims[1] = 4;
    }

    int size_in_bytes = DimsVectorUtils::Count(dims) * data_type_size;
    if (src.GetDeviceType() == DEVICE_NAIVE && dst.GetDeviceType() == DEVICE_CUDA) {
        cudaMemcpy(dst.GetData(), src.GetData(), size_in_bytes, cudaMemcpyHostToDevice);
    } else if (src.GetDeviceType() == DEVICE_CUDA && dst.GetDeviceType() == DEVICE_NAIVE) {
        cudaMemcpy(dst.GetData(), src.GetData(), size_in_bytes, cudaMemcpyDeviceToHost);
    } else if (src.GetDeviceType() == DEVICE_CUDA && dst.GetDeviceType() == DEVICE_CUDA) {
        cudaMemcpy(dst.GetData(), src.GetData(), size_in_bytes, cudaMemcpyDeviceToDevice);
    } else {
        memcpy(dst.GetData(), src.GetData(), size_in_bytes);
    }
    return ret;
}

Status CudaMatConverterAcc::Resize(Mat& src, Mat& dst, ResizeParam param, void* command_queue) {
    Status ret = TNN_OK;

    ret = CheckMatConverterParams(src, dst, true);
    if (ret != TNN_OK)
        return ret;

    int dst_width  = dst.GetWidth();
    int dst_height = dst.GetHeight();
    if (dst_width == 0 || dst_height == 0) {
        return Status(TNNERR_INVALID_INPUT, "dst size is zero");
    }

    if (src.GetMatType() == NCHW_FLOAT) {
        ret = Status(TNNERR_PARAM_ERR, "convert type not support yet");
    } else if ((src.GetMatType() == N8UC4) || (src.GetMatType() == N8UC3) || (src.GetMatType() == NGRAY)) {
        int channel = src.GetChannel();
        if (param.type == INTERP_TYPE_LINEAR) {
            ResizeBilinear((uint8_t*)src.GetData(), (uint8_t*)dst.GetData(), src.GetBatch(), src.GetWidth(),
                src.GetHeight(), dst_width, dst_height, channel);
        } else if(param.type == INTERP_TYPE_NEAREST) {
            ResizeNearest((uint8_t*)src.GetData(), (uint8_t*)dst.GetData(), src.GetBatch(), src.GetWidth(),
                src.GetHeight(), dst_width, dst_height, channel);
        } else {
            return Status(TNNERR_PARAM_ERR, "interpolation type not support yet");
        }
    } else {
        ret = Status(TNNERR_PARAM_ERR, "convert type not support yet");
    }
    return ret;
}

Status CudaMatConverterAcc::Crop(Mat& src, Mat& dst, CropParam param, void* command_queue) {
    Status ret = TNN_OK;

    ret = CheckMatConverterParams(src, dst, true);
    if (ret != TNN_OK)
        return ret;

    if (src.GetMatType() == NGRAY || src.GetMatType() == N8UC3 || src.GetMatType() == N8UC4) {
        auto mat_type = src.GetMatType();
        int channel = mat_type == NGRAY? 1 : (mat_type == N8UC3? 3 : 4);
        uint8_t* src_ptr = (uint8_t*)src.GetData();
        uint8_t* dst_ptr = (uint8_t*)dst.GetData();
        CropRGB(src_ptr, dst_ptr, src.GetBatch(), channel, src.GetWidth(), src.GetHeight(), dst.GetWidth(),
            dst.GetHeight(), param.width, param.height, param.top_left_x, param.top_left_y);
    } else if (src.GetMatType() == NNV21 || src.GetMatType() == NNV12) {
        if (param.top_left_x % 2 || param.top_left_y % 2 || param.width % 2 || param.height % 2) {
            return Status(TNNERR_PARAM_ERR, "corp param can not be odd");
        }
        uint8_t* src_ptr = (uint8_t*)src.GetData();
        uint8_t* dst_ptr = (uint8_t*)dst.GetData();
        CropYUV(src_ptr, dst_ptr, src.GetBatch(), src.GetWidth(), src.GetHeight(), dst.GetWidth(),
            dst.GetHeight(), param.width, param.height, param.top_left_x, param.top_left_y);
    } else {
        return Status(TNNERR_PARAM_ERR, "convert type not support yet");
    }
    return ret;
}

Status CudaMatConverterAcc::WarpAffine(Mat& src, Mat& dst, WarpAffineParam param, void* command_queue) {
    Status ret = TNN_OK;

    ret = CheckMatConverterParams(src, dst, true);
    if (ret != TNN_OK)
        return ret;

    if (param.interp_type == INTERP_TYPE_LINEAR && 
    (param.border_type == BORDER_TYPE_CONSTANT || param.border_type == BORDER_TYPE_TRANSPARENT || param.border_type == BORDER_TYPE_REPLICATE)) {
        if (src.GetMatType() == NGRAY || src.GetMatType() == N8UC3 || src.GetMatType() == N8UC4) {
            int channel = src.GetMatType() == NGRAY ? 1 : (src.GetMatType() == N8UC3 ? 3 : 4);
            uint8_t* src_ptr = (uint8_t*)src.GetData();
            uint8_t* dst_ptr = (uint8_t*)dst.GetData();
            WarpAffineBilinear(src_ptr, src.GetBatch(), channel, src.GetWidth(), src.GetHeight(), dst_ptr, dst.GetWidth(),
                dst.GetHeight(), param.transform, param.border_val, param.border_type, command_queue);
        } else {
            return Status(TNNERR_PARAM_ERR, "convert type not support yet");
        }
    } else if (param.interp_type == INTERP_TYPE_NEAREST && 
    (param.border_type == BORDER_TYPE_CONSTANT || param.border_type == BORDER_TYPE_TRANSPARENT || param.border_type == BORDER_TYPE_REPLICATE)) {
        if (src.GetMatType() == NGRAY || src.GetMatType() == N8UC3 || src.GetMatType() == N8UC4) {
            int channel = src.GetMatType() == NGRAY ? 1 : (src.GetMatType() == N8UC3 ? 3 : 4);
            uint8_t* src_ptr = (uint8_t*)src.GetData();
            uint8_t* dst_ptr = (uint8_t*)dst.GetData();
            WarpAffineNearest(src_ptr, src.GetBatch(), channel, src.GetWidth(), src.GetHeight(), dst_ptr, dst.GetWidth(),
                dst.GetHeight(), param.transform, param.border_val, param.border_type, command_queue);
        } else {
            return Status(TNNERR_PARAM_ERR, "convert type not support yet");
        }
    }

    return ret;
}

Status CudaMatConverterAcc::CvtColor(Mat& src, Mat& dst, ColorConversionType type, void* command_queue) {
    Status ret = TNN_OK;

    ret = CheckMatConverterParams(src, dst, true);
    if (ret != TNN_OK)
        return ret;

    if (type == COLOR_CONVERT_NV12TOBGR) {
        YUVToGRBA((uint8_t*)src.GetData(), (uint8_t*)dst.GetData(), src.GetBatch(), src.GetHeight(), src.GetWidth(), 3, true);
    } else if (type == COLOR_CONVERT_NV21TOBGR) {
        YUVToGRBA((uint8_t*)src.GetData(), (uint8_t*)dst.GetData(), src.GetBatch(), src.GetHeight(), src.GetWidth(), 3, false);
    } else if (type == COLOR_CONVERT_NV12TOBGRA) {
        YUVToGRBA((uint8_t*)src.GetData(), (uint8_t*)dst.GetData(), src.GetBatch(), src.GetHeight(), src.GetWidth(), 4, true);
    } else if (type == COLOR_CONVERT_NV21TOBGRA) {
        YUVToGRBA((uint8_t*)src.GetData(), (uint8_t*)dst.GetData(), src.GetBatch(), src.GetHeight(), src.GetWidth(), 4, false);
    } else if (type == COLOR_CONVERT_BGRTOGRAY) {
        BGRAToGRAY((uint8_t*)src.GetData(), (uint8_t*)dst.GetData(), src.GetBatch(), src.GetHeight(), src.GetWidth(), 3);
    } else if (type == COLOR_CONVERT_BGRATOGRAY) {
        BGRAToGRAY((uint8_t*)src.GetData(), (uint8_t*)dst.GetData(), src.GetBatch(), src.GetHeight(), src.GetWidth(), 4);
    } else {
        return Status(TNNERR_PARAM_ERR, "color conversion type not support yet");
    }

    return ret;
}

Status CudaMatConverterAcc::CopyMakeBorder(Mat& src, Mat& dst, CopyMakeBorderParam param, void* command_queue) {
    Status ret = TNN_OK;

    ret = CheckMatConverterParams(src, dst, true);
    if (ret != TNN_OK)
        return ret;

    uint8_t* src_ptr = (uint8_t*)src.GetData();
    uint8_t* dst_ptr = (uint8_t*)dst.GetData();

    if (src.GetMatType() == NGRAY) {
        CudaCopyMakeBorder(src_ptr, dst_ptr, src.GetBatch(), src.GetWidth(), src.GetHeight(), dst.GetWidth(),
            dst.GetHeight(), 1, param.top, param.bottom, param.left, param.right, uint8_t(param.border_val));
    } else if (src.GetMatType() == N8UC3) {
        CudaCopyMakeBorder(src_ptr, dst_ptr, src.GetBatch(), src.GetWidth(), src.GetHeight(), dst.GetWidth(),
            dst.GetHeight(), 3, param.top, param.bottom, param.left, param.right, uint8_t(param.border_val));
    } else if (src.GetMatType() == N8UC4) {
        CudaCopyMakeBorder(src_ptr, dst_ptr, src.GetBatch(), src.GetWidth(), src.GetHeight(), dst.GetWidth(),
            dst.GetHeight(), 4, param.top, param.bottom, param.left, param.right, uint8_t(param.border_val));
    } else {
        return Status(TNNERR_PARAM_ERR, "CopyMakeBorder mat type not support yet");
    }

    return ret;
}

DECLARE_MAT_CONVERTER_CREATER(Cuda);
REGISTER_MAT_CONVERTER(Cuda, DEVICE_CUDA);

}

