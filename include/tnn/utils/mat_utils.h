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

#ifndef TNN_INCLUDE_TNN_UTILS_MAT_UTILS_H_
#define TNN_INCLUDE_TNN_UTILS_MAT_UTILS_H_

#include "tnn/core/status.h"
#include "tnn/core/mat.h"

namespace TNN_NS {

typedef enum {
    INTERP_TYPE_NEAREST = 0x00,
    INTERP_TYPE_LINEAR  = 0x01,
} PUBLIC InterpType;

typedef enum {
    BORDER_TYPE_CONSTANT = 0x00,
    BORDER_TYPE_REFLECT  = 0x01,
    BORDER_TYPE_EDGE     = 0x02,
} PUBLIC BorderType;

typedef enum {
    COLOR_CONVERT_NV12TOBGR  = 0x00,
    COLOR_CONVERT_NV12TOBGRA = 0x01,
    COLOR_CONVERT_NV21TOBGR  = 0x02,
    COLOR_CONVERT_NV21TOBGRA = 0x03,
    COLOR_CONVERT_BGRTOGRAY  = 0x04,
    COLOR_CONVERT_BGRATOGRAY = 0x05,
} PUBLIC ColorConversionType;

struct PUBLIC ResizeParam {
    float scale_w = 0.0f;
    float scale_h = 0.0f;
    InterpType type = INTERP_TYPE_LINEAR;
};

struct PUBLIC CropParam {
    int top_left_x = 0;
    int top_left_y = 0;
    int width      = 0;
    int height     = 0;
};

struct PUBLIC WarpAffineParam {
    float transform[2][3];
    InterpType interp_type = INTERP_TYPE_NEAREST;
    BorderType border_type = BORDER_TYPE_CONSTANT;
    float border_val       = 0.0f;
};

typedef enum {
    PASTE_TYPE_TOP_LEFT_ALIGN = 0x00,
    PASTE_TYPE_CENTER_ALIGN   = 0x01,
} PUBLIC PasteType;

struct PUBLIC PasteParam {
    PasteType type = PASTE_TYPE_TOP_LEFT_ALIGN;
    int pad_value  = 0;
};

class PUBLIC MatUtils {
public:
    //copy cpu <-> device, cpu<->cpu, device<->device, src and dst dims must be equal.
    static Status Copy(Mat& src, Mat& dst, void* command_queue);

    //src and dst device type must be same. when param scale_w or scale_h is 0, it is computed as
    // (double)dst.GetWidth() / src.GetWidth() or (double)dst.GetHeight() / src.GetHeight().
    static Status Resize(Mat& src, Mat& dst, ResizeParam param, void* command_queue);

    //src and dst device type must be same. when param width or height is 0, it is equal to
    //dst.GetWidth() or dst.GetHeight().
    static Status Crop(Mat& src, Mat& dst, CropParam param, void* command_queue);

    //src and dst device type must be same.
    static Status WarpAffine(Mat& src, Mat& dst, WarpAffineParam param, void* command_queue);

    //src and dst device type must be same.
    static Status CvtColor(Mat& src, Mat& dst, ColorConversionType type, void* command_queue);

    // @brief mat resize and paste to dst
    // @param src  src mat
    // @param dst mat to paste
    // @param param  param to use for resize
    // @param paste_param  param to use for paste
    // @param command_queue  device related command queue
    // @return ret  return val
    static Status ResizeAndPaste(Mat& src, Mat& dst, ResizeParam param, PasteParam paste_param, void* command_queue);

    // @brief mat concat with batch
    // @param src_vec  src mat vector
    // @param dst mat
    // @param command_queue  device related command queue
    // @return ret  return val
    static Status ConcatMatWithBatch(std::vector<Mat>& src_vec, Mat& dst, void* command_queue);

    // @brief get mat data buffer size
    // @param src mat
    // @param byte_size  buffer byte size
    // @return ret  return val 
    static Status GetMatByteSize(Mat& src, int& byte_size);
};

}  // namespace TNN_NS

#endif  // TNN_INCLUDE_TNN_UTILS_MAT_UTILS_H_
