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
#ifndef TNN_SOURCE_TNN_DEVICE_CPU_CPU_MAT_CONVERTER_H_
#define TNN_SOURCE_TNN_DEVICE_CPU_CPU_MAT_CONVERTER_H_
#include "tnn/core/macro.h"
#include "tnn/device/cpu/acc/compute/compute_int8.h"
#include "tnn/utils/blob_converter.h"
#include "tnn/utils/blob_converter_internal.h"
#include "tnn/utils/mat_converter_acc.h"
#include "tnn/device/cpu/cpu_mat_util.h"

namespace TNN_NS {
#define GET_OFFSET_PTR(ptr, offset) (reinterpret_cast<int8_t*>(ptr) + offset)
class CpuMatConverterAcc : public MatConverterAcc {
public:
    CpuMatConverterAcc();
    virtual ~CpuMatConverterAcc();
    virtual Status Copy(Mat& src, Mat& dst, void* command_queue = NULL);
    virtual Status Resize(Mat& src, Mat& dst, ResizeParam param, void* command_queue = NULL);
    virtual Status Crop(Mat& src, Mat& dst, CropParam param, void* command_queue = NULL);
    virtual Status WarpAffine(Mat& src, Mat& dst, WarpAffineParam param, void* command_queue = NULL);
    virtual Status CvtColor(Mat& src, Mat& dst, ColorConversionType type, void* command_queue = NULL);
    virtual Status CopyMakeBorder(Mat& src, Mat& dst, CopyMakeBorderParam param, void* command_queue = NULL);

private:
    void MatMemcpy2D(void* src, void* dst, int width, int height, int src_stride, int dst_stride);
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_CPU_CPU_MAT_CONVERTER_H_
