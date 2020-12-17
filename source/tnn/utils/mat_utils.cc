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

#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/mat_utils.h"
#include "tnn/utils/mat_converter_acc.h"
#include <math.h>

namespace TNN_NS {

#define MAT_CONVERTER_PREPARATION(device_type)                                          \
    if (dst.GetData() == nullptr) {                                                     \
        dst = Mat(dst.GetDeviceType(), dst.GetMatType(), dst.GetDims());                \
    }                                                                                   \
    auto converter = MatConverterManager::Shared()->CreateMatConverterAcc(device_type); \
    if (!converter) {                                                                   \
        return Status(TNNERR_INIT_LAYER, "image converter is nil, check device type");  \
    }

#define CHECK_DST_DATA_NULL                                                             \
    if (dst.GetData() != nullptr) {                                                     \
        return Status(TNNERR_PARAM_ERR, "Incompatible param and dst size.\n "           \
                      "\tSet compatible param and dst size, "                           \
                      "or set dst mat data to null and let tnn infer dst size.");       \
    }

static Status CheckSrcAndDstMat(Mat& src, Mat& dst, bool check_device_type, bool check_mat_type,
                                bool check_src_size) {
    if (check_device_type && (src.GetDeviceType() != dst.GetDeviceType())) {
        return Status(TNNERR_PARAM_ERR, "src and dst DeviceType not equal");
    }

    if (check_mat_type && (src.GetMatType() != dst.GetMatType())) {
        return Status(TNNERR_PARAM_ERR, "src and dst MatType not equal");
    }

    if (check_src_size && (src.GetWidth() <= 0 || src.GetHeight() <= 0)) {
        return Status(TNNERR_INVALID_INPUT, "src size is zero or negnative");
    }

    return TNN_OK;
}

static int GetCvtColorDstChannel(ColorConversionType type) {
    switch (type) {
        case COLOR_CONVERT_BGRTOGRAY:
        case COLOR_CONVERT_BGRATOGRAY:
            return 1;
        case COLOR_CONVERT_NV12TOBGR:
        case COLOR_CONVERT_NV21TOBGR:
            return 3;
        case COLOR_CONVERT_NV12TOBGRA:
        case COLOR_CONVERT_NV21TOBGRA:
            return 4;
        default:
            return Status(TNNERR_PARAM_ERR, "color conversion type not supported");
    }
}

Status MatUtils::Copy(Mat& src, Mat& dst, void* command_queue) {
    auto ret = CheckSrcAndDstMat(src, dst, false, true, true);
    if (ret != TNN_OK) {
        return ret;
    }

    DimsVector src_dims = src.GetDims();
    DimsVector dst_dims = dst.GetDims();
    if (DimsVectorUtils::Equal(src_dims, dst_dims)) {
        DeviceType device_type = DEVICE_NAIVE;
        // get device type
        DeviceType src_dt = src.GetDeviceType();
        DeviceType dst_dt = dst.GetDeviceType();
        if (src_dt == dst_dt) {
            device_type = src_dt;
        } else if (DEVICE_NAIVE == src_dt || DEVICE_ARM == src_dt) {
            device_type = dst_dt;
        } else if (DEVICE_NAIVE == dst_dt || DEVICE_ARM == dst_dt) {
            device_type = src_dt;
        } else {
            return Status(TNNERR_PARAM_ERR, "src and dst DeviceType need be equal or one is device cpu");
        }
        MAT_CONVERTER_PREPARATION(device_type);
        return converter->Copy(src, dst, command_queue);
    } else {
        return Status(TNNERR_PARAM_ERR, "src and dst dims not equal");
    }
}

Status MatUtils::Resize(Mat& src, Mat& dst, ResizeParam param, void* command_queue) {
    auto ret = CheckSrcAndDstMat(src, dst, true, true, true);
    if (ret != TNN_OK) {
        return ret;
    }

    if (param.scale_w > 0 && param.scale_h > 0) {
        int new_h = int(round(param.scale_h * src.GetHeight()));
        int new_w = int(round(param.scale_w * src.GetWidth()));
        if (dst.GetWidth() != new_w || dst.GetHeight() != new_h) {
            CHECK_DST_DATA_NULL;
            // calculate dst size using param scale_h and scale_w
            DimsVector dims = {src.GetBatch(), src.GetChannel(), new_h, new_w};
            dst = Mat(dst.GetDeviceType(), dst.GetMatType(), dims);
        }
    } else {
        if (dst.GetWidth() <= 0 || dst.GetHeight() <= 0) {
            return Status(TNNERR_PARAM_ERR, "both dsize and param scale have zero or negnative value");
        } else {
            param.scale_w = dst.GetWidth() * 1.0 / src.GetWidth();
            param.scale_h = dst.GetHeight() * 1.0 / src.GetHeight();
        }
    }

    MAT_CONVERTER_PREPARATION(src.GetDeviceType());
    return converter->Resize(src, dst, param, command_queue);
}

Status MatUtils::Crop(Mat& src, Mat& dst, CropParam param, void* command_queue) {
    auto ret = CheckSrcAndDstMat(src, dst, true, true, true);
    if (ret != TNN_OK) {
        return ret;
    }

    if (param.width > 0 && param.height > 0) {
        if (dst.GetWidth() != param.width || dst.GetHeight() != param.height) {
            CHECK_DST_DATA_NULL;
            // set dst size by param height and width
            DimsVector dims = {src.GetBatch(), src.GetChannel(), param.height, param.width};
            dst = Mat(dst.GetDeviceType(), dst.GetMatType(), dims);
        }
    } else {
        if (dst.GetWidth() <= 0 || dst.GetHeight() <= 0) {
            return Status(TNNERR_PARAM_ERR, "both dsize and param size have zero or negnative value");
        } else {
            param.width  = dst.GetWidth();
            param.height = dst.GetHeight();
        }
    }

    MAT_CONVERTER_PREPARATION(src.GetDeviceType());
    return converter->Crop(src, dst, param, command_queue);
}

Status MatUtils::WarpAffine(Mat& src, Mat& dst, WarpAffineParam param, void* command_queue) {
    auto ret = CheckSrcAndDstMat(src, dst, true, true, true);
    if (ret != TNN_OK) {
        return ret;
    }

    if (dst.GetData() == nullptr) {
        // set dst size to src size
        dst = Mat(dst.GetDeviceType(), dst.GetMatType(), src.GetDims());
    }

    MAT_CONVERTER_PREPARATION(src.GetDeviceType());
    return converter->WarpAffine(src, dst, param, command_queue);
}

Status MatUtils::CvtColor(Mat& src, Mat& dst, ColorConversionType type, void* command_queue) {
    auto ret = CheckSrcAndDstMat(src, dst, true, false, true);
    if (ret != TNN_OK) {
        return ret;
    }

    if (dst.GetData() == nullptr) {
        // set dst size by src size and cvt type
        DimsVector dims = src.GetDims();
        dims[1] = GetCvtColorDstChannel(type);
        dst = Mat(dst.GetDeviceType(), dst.GetMatType(), dims);
    } else {
        if (dst.GetWidth() < src.GetWidth() || dst.GetHeight() < src.GetHeight() ||
            dst.GetChannel() < GetCvtColorDstChannel(type)) {
            return Status(TNNERR_PARAM_ERR, "cvt color dst size too small");
        }
    }

    MAT_CONVERTER_PREPARATION(src.GetDeviceType());
    return converter->CvtColor(src, dst, type, command_queue);
}

Status MatUtils::CopyMakeBorder(Mat& src, Mat& dst, CopyMakeBorderParam param, void* command_queue) {
    auto ret = CheckSrcAndDstMat(src, dst, true, true, true);
    if (ret != TNN_OK) {
        return ret;
    }

    if (param.top >= 0 && param.bottom >= 0 && param.left >= 0 && param.right >= 0) {
        int new_h = src.GetHeight() + param.top + param.bottom;
        int new_w = src.GetWidth() + param.left + param.right;
        if (dst.GetWidth() != new_w || dst.GetHeight() != new_h) {
            CHECK_DST_DATA_NULL;
            // calculate dst size using param top, bottom, left and right
            DimsVector dims = {src.GetBatch(), src.GetChannel(), new_h, new_w};
            dst = Mat(dst.GetDeviceType(), dst.GetMatType(), dims);
        }
    } else {
        return Status(TNNERR_PARAM_ERR, "border size is negnative");
    }

    MAT_CONVERTER_PREPARATION(src.GetDeviceType());
    return converter->CopyMakeBorder(src, dst, param, command_queue);
}

#undef CHECK_DST_DATA_NULL
#undef MAT_CONVERTER_PREPARATION

}  // namespace TNN_NS
