// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef TNN_SOURCE_TNN_DEVICE_ATLAS_TORCHAIE_BLOB_CONVERTER_H_
#define TNN_SOURCE_TNN_DEVICE_ATLAS_TORCHAIE_BLOB_CONVERTER_H_

#include "tnn/core/macro.h"
#include "tnn/utils/blob_converter.h"
#include "tnn/utils/blob_converter_internal.h"

namespace TNN_NS {

class TorchAieBlobConverterAcc : public BlobConverterAcc {
public:
    TorchAieBlobConverterAcc(Blob* blob);
    virtual ~TorchAieBlobConverterAcc() {};

    virtual Status ConvertToMat(Mat& mat, MatConvertParam param, void* command_queue = NULL);
    virtual Status ConvertToMatAsync(Mat& mat, MatConvertParam param, void* command_queue = NULL);

    virtual Status ConvertFromMat(Mat& mat, MatConvertParam param, void* command_queue = NULL);
    virtual Status ConvertFromMatAsync(Mat& mat, MatConvertParam param, void* command_queue = NULL);
private:
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_ATLAS_TORCHAIE_BLOB_CONVERTER_H_
