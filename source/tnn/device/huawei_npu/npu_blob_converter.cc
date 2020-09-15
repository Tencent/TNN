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
#ifndef TNN_SOURCE_TNN_DEVICE_HUAWEI_NPU_NPU_BLOB_CONVERTER_CC_
#define TNN_SOURCE_TNN_DEVICE_HUAWEI_NPU_NPU_BLOB_CONVERTER_CC_
#include "tnn/core/macro.h"
#include "tnn/device/cpu/cpu_blob_converter.h"
#include "tnn/utils/blob_converter.h"

namespace TNN_NS {

class NpuBlobConverterAcc : public CpuBlobConverterAcc {
public:
    NpuBlobConverterAcc(Blob *blob) : CpuBlobConverterAcc(blob) {}
    ~NpuBlobConverterAcc() {}
    virtual Status ConvertToMat(Mat &image, MatConvertParam param, void *command_queue = NULL) {
        return CpuBlobConverterAcc::ConvertToMat(image, param, command_queue);
    }

    virtual Status ConvertToMatAsync(Mat &image, MatConvertParam param, void *command_queue = NULL) {
        return CpuBlobConverterAcc::ConvertToMatAsync(image, param, command_queue);
    }

    virtual Status ConvertFromMat(Mat &image, MatConvertParam param, void *command_queue = NULL) {
        return CpuBlobConverterAcc::ConvertFromMat(image, param, command_queue);
    }
    virtual Status ConvertFromMatAsync(Mat &image, MatConvertParam param, void *command_queue = NULL) {
        return CpuBlobConverterAcc::ConvertFromMatAsync(image, param, command_queue);
    }
};

DECLARE_BLOB_CONVERTER_CREATER(Npu);
REGISTER_BLOB_CONVERTER(Npu, DEVICE_HUAWEI_NPU);
}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_HUAWEI_NPU_NPU_BLOB_CONVERTER_CC_
