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

#include "tnn/device/cpu/cpu_mat_converter.h"

#include <algorithm>
#include <cstring>

#include "tnn/core/blob_int8.h"
#include "tnn/core/macro.h"
#include "tnn/utils/naive_compute.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/bfp16_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/mat_converter_utils.h"

namespace TNN_NS {

CpuMatConverterAcc::CpuMatConverterAcc() : MatConverterAcc() {}
CpuMatConverterAcc::~CpuMatConverterAcc() {}

Status CpuMatConverterAcc::Copy(Mat& src, Mat& dst, void* command_queue) {
    Status ret            = TNN_OK;

    ret = CheckMatConverterParams(src, dst, false);
    if (ret != TNN_OK)
        return ret;

    MatType mat_type   = src.GetMatType();
    int data_type_size = 1;
    DimsVector dims    = src.GetDims();
    if (mat_type == NCHW_FLOAT) {
        data_type_size = sizeof(float);
    } else if (mat_type == N8UC4) {
        //special for 8UC4, blob channel <= 4.
        dims[1] = 4;
    }
    int size_in_bytes = DimsVectorUtils::Count(dims) * data_type_size;
    memcpy(dst.GetData(), src.GetData(), size_in_bytes);
    return ret; 
}

Status CpuMatConverterAcc::Resize(Mat& src, Mat& dst, ResizeParam param, void* command_queue) {
    Status ret            = TNN_OK;

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
            for (int batch = 0; batch < src.GetBatch(); batch++)
            {
                uint8_t* src_ptr = (uint8_t*)src.GetData() + batch * src.GetWidth() * src.GetHeight() * channel;
                uint8_t* dst_ptr = (uint8_t*)dst.GetData() + batch * dst_width * dst_height * channel;
                ResizeBilinear(src_ptr, src.GetWidth(), src.GetHeight(),
                               dst_ptr, dst_width, dst_height, channel);
            } 
        } else if(param.type == INTERP_TYPE_NEAREST) {
            ResizeNearest((uint8_t*)src.GetData(), src.GetBatch(), src.GetWidth(), src.GetHeight(),
            (uint8_t*)dst.GetData(), dst_width, dst_height, channel);
        } else {
            return Status(TNNERR_PARAM_ERR, "interpolation type not support yet");
        }
    } else if (src.GetMatType() == RESERVED_BFP16_TEST) {
        ret = Status(TNNERR_PARAM_ERR, "convert type not support yet");
    } else {
        ret = Status(TNNERR_PARAM_ERR, "convert type not support yet");
    }
    return ret;
}

Status CpuMatConverterAcc::Crop(Mat& src, Mat& dst, CropParam param, void* command_queue) {
    Status ret            = TNN_OK;

    ret = CheckMatConverterParams(src, dst, true);
    if (ret != TNN_OK)
        return ret;

    if (src.GetMatType() == NGRAY || src.GetMatType() == N8UC3 || src.GetMatType() == N8UC4) {
        auto mat_type = src.GetMatType();
        int channel = mat_type == NGRAY? 1 : (mat_type == N8UC3? 3 : 4);
        for (int batch = 0; batch < src.GetBatch(); batch++)
        {
            auto src_ptr = GET_OFFSET_PTR((uint8_t*)src.GetData() + batch * src.GetWidth() * src.GetHeight() * channel, (param.top_left_x + param.top_left_y * src.GetWidth()) * channel);
            auto dst_ptr = GET_OFFSET_PTR((uint8_t*)dst.GetData() + batch * dst.GetWidth() * dst.GetHeight() * channel, 0);
            MatMemcpy2D(src_ptr, dst_ptr, param.width * channel, param.height, src.GetWidth() * channel, dst.GetWidth() * channel);
        } 
    } else if (src.GetMatType() == NNV21 || src.GetMatType() == NNV12) {
        if (param.top_left_x % 2 || param.top_left_y % 2 || param.width % 2 || param.height % 2) {
            return Status(TNNERR_PARAM_ERR, "corp param can not be odd");
        }
        // crop y
        auto src_ptr = GET_OFFSET_PTR(src.GetData(), param.top_left_x + param.top_left_y * param.width);
        auto dst_ptr = GET_OFFSET_PTR(dst.GetData(), 0);
        MatMemcpy2D(src_ptr, dst_ptr, param.width, param.height, src.GetWidth(), dst.GetWidth());
        // crop uv
        src_ptr = GET_OFFSET_PTR(
            src.GetData(), src.GetWidth() * src.GetHeight() + param.top_left_x + param.top_left_y * src.GetWidth() / 2);
        dst_ptr = GET_OFFSET_PTR(dst.GetData(), dst.GetWidth() * dst.GetHeight());
        MatMemcpy2D(src_ptr, dst_ptr, param.width, param.height / 2, src.GetWidth(), dst.GetWidth());
    } else {
        return Status(TNNERR_PARAM_ERR, "convert type not support yet");
    }

    return ret;
}

Status CpuMatConverterAcc::WarpAffine(Mat& src, Mat& dst, WarpAffineParam param, void* command_queue) {
    //LOGE("cpu mat converter warp affine start, mat type: %d, interp type: %d\n", src.GetMatType(), param.interp_type);
    Status ret            = TNN_OK;

    ret = CheckMatConverterParams(src, dst, true);
    if (ret != TNN_OK)
        return ret;

    if (src.GetMatType() == NGRAY || src.GetMatType() == N8UC3 || src.GetMatType() == N8UC4) {
        auto mat_type = src.GetMatType();
        int channel = mat_type == NGRAY? 1 : (mat_type == N8UC3? 3 : 4);
        if (param.interp_type == INTERP_TYPE_LINEAR && param.border_type == BORDER_TYPE_CONSTANT) {
            for (int batch = 0; batch < src.GetDims()[0]; batch++)
            {
                uint8_t* src_ptr = (uint8_t*)src.GetData() + batch * src.GetWidth() * src.GetHeight() * channel;
                uint8_t* dst_ptr = (uint8_t*)dst.GetData() + batch * dst.GetWidth() * dst.GetHeight() * channel;
                WarpAffineBilinear(src_ptr, src.GetWidth(), src.GetHeight(), channel,
                                   dst_ptr, dst.GetWidth(), dst.GetHeight(),
                                   param.transform, param.border_val);
            }
        } else if (param.interp_type == INTERP_TYPE_NEAREST && param.border_type == BORDER_TYPE_CONSTANT) {
            for (int batch = 0; batch < src.GetDims()[0]; batch++)
            {
                uint8_t* src_ptr = (uint8_t*)src.GetData() + batch * src.GetWidth() * src.GetHeight() * channel;
                uint8_t* dst_ptr = (uint8_t*)dst.GetData() + batch * dst.GetWidth() * dst.GetHeight() * channel;
                WarpAffineNearest(src_ptr, src.GetWidth(), src.GetHeight(), channel,
                                  dst_ptr, dst.GetWidth(), dst.GetHeight(),
                                  param.transform, param.border_val);
            }
        } else {
            return Status(TNNERR_PARAM_ERR, "warpaffine type not support yet");
        }
    } else {
        return Status(TNNERR_PARAM_ERR, "convert type not support yet");
    }

    return ret;
}

Status CpuMatConverterAcc::CvtColor(Mat& src, Mat& dst, ColorConversionType type, void* command_queue) {
    Status ret = TNN_OK;

    ret = CheckMatConverterParams(src, dst, true);
    if (ret != TNN_OK)
        return ret;

    if (type == COLOR_CONVERT_NV12TOBGR) {
        YUVToBGR((uint8_t*)src.GetData(), (uint8_t*)dst.GetData(), src.GetBatch()*src.GetHeight(), src.GetWidth(), true);
    } else if (type == COLOR_CONVERT_NV21TOBGR) {
        YUVToBGR((uint8_t*)src.GetData(), (uint8_t*)dst.GetData(), src.GetBatch()*src.GetHeight(), src.GetWidth(), false);
    } else if (type == COLOR_CONVERT_NV12TOBGRA) {
        YUVToBGRA((uint8_t*)src.GetData(), (uint8_t*)dst.GetData(), src.GetBatch()*src.GetHeight(), src.GetWidth(), true);
    } else if (type == COLOR_CONVERT_NV21TOBGRA) {
        YUVToBGRA((uint8_t*)src.GetData(), (uint8_t*)dst.GetData(), src.GetBatch()*src.GetHeight(), src.GetWidth(), false);
    } else if (type == COLOR_CONVERT_BGRTOGRAY) {
        BGROrBGRAToGray((uint8_t*)src.GetData(), (uint8_t*)dst.GetData(), src.GetBatch()*src.GetHeight(), src.GetWidth(), 3);
    } else if (type == COLOR_CONVERT_BGRATOGRAY) {
        BGROrBGRAToGray((uint8_t*)src.GetData(), (uint8_t*)dst.GetData(), src.GetBatch()*src.GetHeight(), src.GetWidth(), 4);
    } else {
        return Status(TNNERR_PARAM_ERR, "color conversion type not support yet");
    }

    return ret;
}

Status CpuMatConverterAcc::CopyMakeBorder(Mat& src, Mat& dst, CopyMakeBorderParam param, void* command_queue) {
    Status ret            = TNN_OK;

    ret = CheckMatConverterParams(src, dst, true);
    if (ret != TNN_OK)
        return ret;

    if (src.GetMatType() == NGRAY || src.GetMatType() == N8UC3 || src.GetMatType() == N8UC4) {
        if (param.border_type == BORDER_TYPE_CONSTANT) {
            auto mat_type = src.GetMatType();
            int channel = mat_type == NGRAY? 1 : (mat_type == N8UC3? 3 : 4);
            for (int i = 0; i < DimsVectorUtils::Count(dst.GetDims()); ++i) {
                ((uint8_t*)dst.GetData())[i] = int(param.border_val);
            }
            for (int batch = 0; batch < src.GetBatch(); batch++)
            {
                auto src_ptr = GET_OFFSET_PTR((uint8_t*)src.GetData() + batch * src.GetWidth() * src.GetHeight() * channel, 0);
                auto dst_ptr = GET_OFFSET_PTR((uint8_t*)dst.GetData() + batch * dst.GetWidth() * dst.GetHeight() * channel,
                                              (param.left + param.top * dst.GetWidth()) * channel);
                MatMemcpy2D(src_ptr, dst_ptr, src.GetWidth() * channel, src.GetHeight(), src.GetWidth() * channel, dst.GetWidth() * channel);
            }
        } else {
            return Status(TNNERR_PARAM_ERR, "CopyMakeBorder border type not support yet");
        }
    } else {
        return Status(TNNERR_PARAM_ERR, "CopyMakeBorder mat type not support yet");
    }

    return ret;
}

void CpuMatConverterAcc::MatMemcpy2D(void* src, void* dst, int width, int height, int src_stride, int dst_stride) {
    auto src_ptr = reinterpret_cast<uint8_t*>(src);
    auto dst_ptr = reinterpret_cast<uint8_t*>(dst);

    for (int h = 0; h < height; h++) {
        memcpy(dst_ptr, src_ptr, width);
        src_ptr += src_stride;
        dst_ptr += dst_stride;
    }

}

DECLARE_MAT_CONVERTER_CREATER(Cpu);
REGISTER_MAT_CONVERTER(Cpu, DEVICE_NAIVE);

}
