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

typedef Status (*ArmBlobConvertFunc)(Mat& image,
                                     char* handle_ptr,
                                     const MatConvertParam& param,
                                     const DimsVector& dims,
                                     const int hw,
                                     const int c_r4,
                                     std::vector<float>& fused_int8_scale,
                                     std::vector<float>& fused_int8_bias);

typedef enum {
    CVT_DIR_MAT2BLOB = 0,
    CVT_DIR_BLOB2MAT = 1
} BlobConvertDirection;

class ArmBlobConverterAcc : public BlobConverterAcc {
public:
    ArmBlobConverterAcc(Blob* blob);
    virtual ~ArmBlobConverterAcc();

    virtual Status ConvertToMat(Mat& image, MatConvertParam param, void* command_queue = NULL);
    virtual Status ConvertToMatAsync(Mat& image, MatConvertParam param, void* command_queue = NULL);

    virtual Status ConvertFromMat(Mat& image, MatConvertParam param, void* command_queue = NULL);
    virtual Status ConvertFromMatAsync(Mat& image, MatConvertParam param, void* command_queue = NULL);

    static Status RegisterBlobConvertFunc(MatType mat_type, DataType data_type, BlobConvertDirection cvt_dir,
                                          ArmBlobConvertFunc cvt_func);

private:
    Status ReverseInputImageChannel(Mat& image, const BlobDesc& desc, const DimsVector& dims, const int hw);
    Status ReverseOutImageChannel(Mat& image, const BlobDesc& desc, const DimsVector& dims, const int hw);

    std::vector<float> fused_int8_scale;
    std::vector<float> fused_int8_bias;
    ArmBlobConvertFunc cvt_func_;

    static Status GetBlobConvertFunc(MatType mat_type, DataType data_type, BlobConvertDirection cvt_dir,
                                     ArmBlobConvertFunc& cvt_func);
    static std::string GetUniqueBlobConvertKey(MatType mat_type, DataType data_type, BlobConvertDirection cvt_dir);
    static std::map<std::string, ArmBlobConvertFunc>& GetBlobConvertFuncMap();
};

class ArmBlobConvertFuncRegister {
public:
    explicit ArmBlobConvertFuncRegister(MatType mat_type, DataType data_type, BlobConvertDirection cvt_dir,
                                        ArmBlobConvertFunc cvt_func) {
        ArmBlobConverterAcc::RegisterBlobConvertFunc(mat_type, data_type, cvt_dir, cvt_func);
    }
};

#define REGISTER_ARM_BLOB_CONVERT_FUNC(mat_type, data_type, cvt_dir, cvt_func)                                  \
    ArmBlobConvertFuncRegister g_arm_##mat_type##_##data_type##_##cvt_dir##_register(mat_type, data_type,       \
                                                                                     cvt_dir, cvt_func);

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_ARM_ARM_IMAGE_CONVERTER_H_
