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

#include "tnn/utils/mat_converter.h"
#include "tnn/utils/mat_utils.h"
#include "tnn/utils/mat_converter_internal.h"

namespace TNN_NS {

class MetalMatConverterAcc : public MatConverterAcc {
public:
    virtual Status Copy(Mat& src, Mat& dst, void* command_queue = NULL);
    virtual Status Resize(Mat& src, Mat& dst, ResizeParam param, void* command_queue = NULL);
    virtual Status Crop(Mat& src, Mat& dst, CropParam param, void* command_queue = NULL);
    virtual Status WarpAffine(Mat& src, Mat& dst, WarpAffineParam param, void* command_queue = NULL);
};

Status MetalMatConverterAcc::Copy(Mat& src, Mat& dst, void* command_queue) {
    return TNN_OK;
}

Status MetalMatConverterAcc::Resize(Mat& src, Mat& dst, ResizeParam param, void* command_queue) {
    return TNN_OK;
}

Status MetalMatConverterAcc::Crop(Mat& src, Mat& dst, CropParam param, void* command_queue) {
    return TNN_OK;
}

Status MetalMatConverterAcc::WarpAffine(Mat& src, Mat& dst, WarpAffineParam param, void* command_queue) {
    return TNN_OK;
}

DECLARE_MAT_CONVERTER_CREATER(Metal);
REGISTER_MAT_CONVERTER(Metal, DEVICE_METAL);

}  // namespace TNN_NS