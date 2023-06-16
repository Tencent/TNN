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

#ifndef TNN_SOURCE_TNN_DEVICE_ATLAS_ATLAS_MAT_CONVERTER_H_
#define TNN_SOURCE_TNN_DEVICE_ATLAS_ATLAS_MAT_CONVERTER_H_

#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"
#include "tnn/core/macro.h"
#include "tnn/utils/mat_converter_acc.h"
#include "tnn/utils/mat_utils.h"

namespace TNN_NS {

class AtlasMatConverterAcc : public MatConverterAcc {
public:
    AtlasMatConverterAcc();
    virtual ~AtlasMatConverterAcc();
    virtual Status Copy(Mat& src, Mat& dst, void* command_queue = NULL) override;
    virtual Status Resize(Mat& src, Mat& dst, ResizeParam param, void* command_queue = NULL) override;
    virtual Status Crop(Mat& src, Mat& dst, CropParam param, void* command_queue = NULL) override;
    virtual Status WarpAffine(Mat& src, Mat& dst, WarpAffineParam param, void* command_queue = NULL) override;
    virtual Status CvtColor(Mat& src, Mat& dst, ColorConversionType type, void* command_queue = NULL) override;
    virtual Status CopyMakeBorder(Mat& src, Mat& dst, CopyMakeBorderParam param, void* command_queue = NULL) override;

private:
    Status PrepareInput(Mat& mat);
    Status PrepareOutput(Mat& mat, int pad_value = 0);
    Status ProcessOutput(Mat& mat);

    Status GetAlignedBufferSize(Mat& mat, int width_align_to, int height_align_to, int& buffer_size, int& width_aligned,
                                int& height_aligned);

    Status MallocDeviceMemory(void** buffer, int& size, int desired_size);

    Status CopyFromHostToDeviceAligned(Mat& src, void* dst, int width_align_to, int height_align_to);

    int GetWidthStride(MatType mat_type, int width);

    CropParam ProcessCropParam(CropParam param);

    Status MatCopyAsync(Mat& dst, Mat& src, int dst_offset, void* stream);

private:
    bool init_success_                     = false;
    acldvppChannelDesc* dvpp_channel_desc_ = nullptr;
    acldvppPicDesc* input_desc_            = nullptr;
    acldvppPicDesc* output_desc_           = nullptr;

    void* dvpp_input_buffer_      = nullptr;
    void* dvpp_input_buffer_ptr_  = nullptr;
    int input_buffer_size_        = 0;
    void* dvpp_output_buffer_     = nullptr;
    void* dvpp_output_buffer_ptr_ = nullptr;
    int output_buffer_size_       = 0;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_ATLAS_ATLAS_MAT_CONVERTER_H_
