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
#ifndef TNN_SOURCE_TNN_UTILS_BLOB_CONVERTER_DEFAULT_H_
#define TNN_SOURCE_TNN_UTILS_BLOB_CONVERTER_DEFAULT_H_

#include "tnn/core/macro.h"
#include "tnn/utils/blob_converter.h"
#include "tnn/utils/blob_converter_internal.h"

namespace TNN_NS {

class DefaultBlobConverterAcc : public BlobConverterAcc {
public:
    DefaultBlobConverterAcc(Blob* blob);
    virtual ~DefaultBlobConverterAcc();

    virtual Status ConvertToMat(Mat& image, MatConvertParam param, void* command_queue = NULL);
    virtual Status ConvertToMatAsync(Mat& image, MatConvertParam param, void* command_queue = NULL);

    virtual Status ConvertFromMat(Mat& image, MatConvertParam param, void* command_queue = NULL);
    virtual Status ConvertFromMatAsync(Mat& image, MatConvertParam param, void* command_queue = NULL);

private:
    Status ConvertFromMatFunc(Mat& image, float* blob_data, MatConvertParam& param, BlobDesc& desc,
                              const DimsVector& dims, const int hw);
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_UTILS_BLOB_CONVERTER_DEFAULT_H_
