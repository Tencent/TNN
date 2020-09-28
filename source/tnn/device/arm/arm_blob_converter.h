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

#ifndef TNN_SOURCE_TNN_DEVICE_ARM_ARM_MAT_CONVERTER_H_
#define TNN_SOURCE_TNN_DEVICE_ARM_ARM_MAT_CONVERTER_H_

#include "tnn/core/macro.h"
#include "tnn/device/arm/arm_util.h"
#include "tnn/utils/blob_converter.h"
#include "tnn/utils/blob_converter_internal.h"

namespace TNN_NS {

class ArmBlobConverterAcc : public BlobConverterAcc {
public:
    ArmBlobConverterAcc(Blob* blob);
    virtual ~ArmBlobConverterAcc();

    virtual Status ConvertToMat(Mat& image, MatConvertParam param, void* command_queue = NULL);
    virtual Status ConvertToMatAsync(Mat& image, MatConvertParam param, void* command_queue = NULL);

    virtual Status ConvertFromMat(Mat& image, MatConvertParam param, void* command_queue = NULL);
    virtual Status ConvertFromMatAsync(Mat& image, MatConvertParam param, void* command_queue = NULL);

private:
    Status ReverseInputImageChannel(Mat& image, const BlobDesc& desc, const DimsVector& dims, const int hw);
    Status ReverseOutImageChannel(Mat& image, const BlobDesc& desc, const DimsVector& dims, const int hw);
    void ConvertImageToBlob(Mat& image, char *handle_ptr,
                            const BlobDesc& desc, const DimsVector& dims, const int hw,
                            MatConvertParam& param,
                            std::vector<float>& fused_int8_scale,
                            std::vector<float>& fused_int8_bias);
    void ConvertYuvImageToBlob(Mat& image, char *handle_ptr,
                               const BlobDesc& desc, const DimsVector& dims, const int hw,
                               MatConvertParam& param,
                               std::vector<float>& fused_int8_scale,
                               std::vector<float>& fused_int8_bias);
    Status ConvertFloatMatToBlob(Mat& image, char *handle_ptr,
                                 const BlobDesc& desc, const DimsVector& dims, const int hw,
                                 const int c_r4,
                                 MatConvertParam& param,
                                 std::vector<float>& fused_int8_scale,
                                 std::vector<float>& fused_int8_bias);
    Status ConvertBlobToFloatMat(Mat& image, char *handle_ptr,
                                 const DimsVector& dims, const int hw,
                                 const int c_r4, MatConvertParam& param,
                                 std::vector<float>& fused_int8_scale);
    std::vector<float> fused_int8_scale;
    std::vector<float> fused_int8_bias;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_ARM_ARM_IMAGE_CONVERTER_H_
