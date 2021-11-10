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

#ifndef TNN_INCLUDE_TNN_UTILS_BLOB_CONVERTER_H_
#define TNN_INCLUDE_TNN_UTILS_BLOB_CONVERTER_H_

#include <memory>
#include <vector>

#include "tnn/core/blob.h"
#include "tnn/core/common.h"
#include "tnn/core/mat.h"
#include "tnn/core/macro.h"
#include "tnn/core/status.h"

#pragma warning(push)
#pragma warning(disable : 4251)

namespace TNN_NS {

//formular: y = scale*x + bias
struct PUBLIC MatConvertParam {
    std::vector<float> scale = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> bias  = {0.0f, 0.0f, 0.0f, 0.0f};
    bool reverse_channel     = false;
};

class BlobConverterAcc;
class PUBLIC BlobConverter {
public:
    explicit BlobConverter(Blob* blob);
    Status ConvertToMat(Mat& image, MatConvertParam param, void* command_queue);
    Status ConvertFromMat(Mat& image, MatConvertParam param, void* command_queue);

    Status ConvertToMatAsync(Mat& image, MatConvertParam param, void* command_queue);
    Status ConvertFromMatAsync(Mat& image, MatConvertParam param, void* command_queue);

private:
    Blob* blob_ = nullptr;
    std::shared_ptr<BlobConverterAcc> impl_ = nullptr;

    Status CheckScaleBiasInParam(Mat& image, MatConvertParam& param, bool convert_to_mat);
    bool NeedDoScaleBias(MatConvertParam &param);
};

}  // namespace TNN_NS

#pragma warning(pop)

#endif  // TNN_INCLUDE_TNN_UTILS_BLOB_CONVERTER_H_
