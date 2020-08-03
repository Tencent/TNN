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

#include "tnn/device/arm/arm_mat_converter.h"

#include "tnn/device/arm/arm_mat_util.h"
#include "tnn/device/arm/arm_util.h"

namespace TNN_NS {

Status ArmMatConverterAcc::Copy(Mat& src, Mat& dst, void* command_queue) {
    Status ret = TNN_OK;
    return ret;
}

Status ArmMatConverterAcc::Resize(Mat& src, Mat& dst, ResizeParam param, void* command_queue) {
    Status ret = TNN_OK;

    if (src.GetData() == nullptr) {
        return Status(TNNERR_NULL_PARAM, "input mat is null");
    }

    if (src.GetDeviceType() != dst.GetDeviceType()) {
        return Status(TNNERR_PARAM_ERR, "src and dst mat type must be same");
    }

    if (dst.GetData() == nullptr) {
        dst = Mat(dst.GetDeviceType(), dst.GetMatType(), dst.GetDims());
    }

    if (src.GetMatType() == NGRAY) {
        if (param.type == INTERP_TYPE_LINEAR) {
            resize_bilinear_c1((uint8_t*)src.GetData(), src.GetWidth(), src.GetHeight(),
                               (uint8_t*)dst.GetData(), dst.GetWidth(), dst.GetHeight());
        } else {
            return Status(TNNERR_PARAM_ERR, "interpolation type not support yet");
        }
    } else if (src.GetMatType() == N8UC3) {
        // resize_bilinear_c1(src.GetData(), src.GetWidth(), src.GetHeight(), src.GetWidth() * 3, dst.GetData(),
        //                    dst.GetWidth(), dst.GetHeight(), dst.GetWidth() * 3);
    } else if (src.GetMatType() == N8UC4) {
        // resize_bilinear_c1(src.GetData(), src.GetWidth(), src.GetHeight(), src.GetWidth() * 4, dst.GetData(),
        //                    dst.GetWidth(), dst.GetHeight(), dst.GetWidth() * 4);
    } else if (src.GetMatType() == NNV21 || src.GetMatType() == NNV12) {
        // resize_bilinear_yuv420sp(src.GetData(), src.GetWidth(), src.GetHeight(), dst.GetData(), dst.GetWidth(),
        //                          dst.GetHeight());
    } else {
        return Status(TNNERR_PARAM_ERR, "convert type not support yet");
    }

    return ret;
}

Status ArmMatConverterAcc::Crop(Mat& src, Mat& dst, CropParam param, void* command_queue) {
    Status ret = TNN_OK;

    if (src.GetData() == nullptr) {
        return Status(TNNERR_NULL_PARAM, "input mat is null");
    }

    if (src.GetDeviceType() != dst.GetDeviceType()) {
        return Status(TNNERR_PARAM_ERR, "src and dst mat type must be same");
    }

    if (dst.GetData() == nullptr) {
        dst = Mat(dst.GetDeviceType(), dst.GetMatType(), dst.GetDims());
    }

    if (src.GetMatType() == NGRAY) {
        // element size 1
        auto src_ptr = GET_OFFSET_PTR(src.GetData(), param.top_left_x + param.top_left_y * src.GetWidth());
        auto dst_ptr = GET_OFFSET_PTR(dst.GetData(), 0);
        mat_memcpy_2d(src_ptr, dst_ptr, param.width, param.height, src.GetWidth(), dst.GetWidth());
    } else if (src.GetMatType() == N8UC3) {
        // element size 3
        auto src_ptr = GET_OFFSET_PTR(src.GetData(), (param.top_left_x + param.top_left_y * src.GetWidth()) * 3);
        auto dst_ptr = GET_OFFSET_PTR(dst.GetData(), 0);
        mat_memcpy_2d(src_ptr, dst_ptr, param.width * 3, param.height, src.GetWidth() * 3, dst.GetWidth() * 3);
    } else if (src.GetMatType() == N8UC4) {
        // element size 4
        auto src_ptr = GET_OFFSET_PTR(src.GetData(), (param.top_left_x + param.top_left_y * src.GetWidth()) * 4);
        auto dst_ptr = GET_OFFSET_PTR(dst.GetData(), 0);
        mat_memcpy_2d(src_ptr, dst_ptr, param.width * 4, param.height, src.GetWidth() * 4, dst.GetWidth() * 4);
    } else if (src.GetMatType() == NNV21 || src.GetMatType() == NNV12) {
        if (param.top_left_x % 2 || param.top_left_y % 2 || param.width % 2 || param.height % 2) {
            return Status(TNNERR_PARAM_ERR, "corp param can not be odd");
        }
        // crop y
        auto src_ptr = GET_OFFSET_PTR(src.GetData(), param.top_left_x + param.top_left_y * src.GetWidth());
        auto dst_ptr = GET_OFFSET_PTR(dst.GetData(), 0);
        mat_memcpy_2d(src_ptr, dst_ptr, param.width, param.height, src.GetWidth(), dst.GetWidth());
        // crop uv
        src_ptr = GET_OFFSET_PTR(
            src.GetData(), src.GetWidth() * src.GetHeight() + param.top_left_x + param.top_left_y * src.GetWidth() / 2);
        dst_ptr = GET_OFFSET_PTR(dst.GetData(), dst.GetWidth() * dst.GetHeight());
        mat_memcpy_2d(src_ptr, dst_ptr, param.width, param.height / 2, src.GetWidth(), dst.GetWidth());
    } else {
        return Status(TNNERR_PARAM_ERR, "convert type not support yet");
    }

    return ret;
}

Status ArmMatConverterAcc::WarpAffine(Mat& src, Mat& dst, WarpAffineParam param, void* command_queue) {
    Status ret = TNN_OK;

    if (src.GetData() == nullptr) {
        return Status(TNNERR_NULL_PARAM, "input mat is null");
    }

    if (src.GetDeviceType() != dst.GetDeviceType()) {
        return Status(TNNERR_PARAM_ERR, "src and dst mat type must be same");
    }

    if (dst.GetData() == nullptr) {
        dst = Mat(dst.GetDeviceType(), dst.GetMatType(), dst.GetDims());
    }

    if (src.GetMatType() == NGRAY) {
        // warpaffine_bilinear_c1(src.GetData(), src.GetWidth(), src.GetHeight(), dst.GetData(),
        //                    dst.GetWidth(), dst.GetHeight(), param.transform);
    } else if (src.GetMatType() == N8UC3) {
        // warpaffine_bilinear_c3(src.GetData(), src.GetWidth(), src.GetHeight(), dst.GetData(),
        //                    dst.GetWidth(), dst.GetHeight(), param.transform);
    } else {
        return Status(TNNERR_PARAM_ERR, "convert type not support yet");
    }

    return ret;
}

DECLARE_MAT_CONVERTER_CREATER(Arm);
REGISTER_MAT_CONVERTER(Arm, DEVICE_ARM);

}  // namespace TNN_NS
