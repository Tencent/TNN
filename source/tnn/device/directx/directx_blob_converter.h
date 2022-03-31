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

#ifndef TNN_SOURCE_TNN_DEVICE_DIRECTX_DIRECTX_BLOB_CONVERTER_H_
#define TNN_SOURCE_TNN_DEVICE_DIRECTX_DIRECTX_BLOB_CONVERTER_H_

#include "tnn/core/macro.h"
#include "tnn/utils/blob_converter_default.h"
#include "tnn/utils/blob_converter.h"
#include "tnn/device/directx/directx_memory.h"
#include "tnn/device/directx/directx_util.h"

namespace TNN_NS {
namespace directx {

typedef Status (*DirectXBlobConvertFunc)(Mat& image,
                                        Blob * blob,
                                        const MatConvertParam& param,
                                        void * command_queue);

typedef enum {
    CVT_DIR_MAT2BLOB = 0,
    CVT_DIR_BLOB2MAT = 1
} BlobConvertDirection;

class DirectXBlobConverterAcc : public DefaultBlobConverterAcc {
public:
    DirectXBlobConverterAcc(Blob *blob) : DefaultBlobConverterAcc(blob) {}
    ~DirectXBlobConverterAcc() {}

    virtual Status ConvertToMat(Mat& image, MatConvertParam param, void* command_queue = NULL) override;
    virtual Status ConvertToMatAsync(Mat& image, MatConvertParam param, void* command_queue = NULL) override;

    virtual Status ConvertFromMat(Mat& image, MatConvertParam param, void* command_queue = NULL) override;
    virtual Status ConvertFromMatAsync(Mat& image, MatConvertParam param, void* command_queue = NULL) override;

    static Status RegisterBlobConvertFunc(MatType mat_type, DataType data_type, BlobConvertDirection cvt_dir,
                                          DirectXBlobConvertFunc cvt_func);

private:
    std::vector<float> fused_int8_scale;
    std::vector<float> fused_int8_bias;
    DirectXBlobConvertFunc cvt_func_ = nullptr;

    static Status GetBlobConvertFunc(MatType mat_type, DataType data_type, BlobConvertDirection cvt_dir,
                                     DirectXBlobConvertFunc& cvt_func);
    static std::string GetUniqueBlobConvertKey(MatType mat_type, DataType data_type, BlobConvertDirection cvt_dir);
    static std::map<std::string, DirectXBlobConvertFunc>& GetBlobConvertFuncMap();
};

class DirectXBlobConvertFuncRegister {
public:
    explicit DirectXBlobConvertFuncRegister(MatType mat_type, DataType data_type, BlobConvertDirection cvt_dir,
                                        DirectXBlobConvertFunc cvt_func) {
        DirectXBlobConverterAcc::RegisterBlobConvertFunc(mat_type, data_type, cvt_dir, cvt_func);
    }
};

#define REGISTER_DIRECTX_BLOB_CONVERT_FUNC(mat_type, data_type, cvt_dir, cvt_func)                                  \
    DirectXBlobConvertFuncRegister g_arm_##mat_type##_##data_type##_##cvt_dir##_register(mat_type, data_type,       \
                                                                                     cvt_dir, cvt_func);

}
}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_DIRECTX_DIRECTX_BLOB_CONVERTER_H_
