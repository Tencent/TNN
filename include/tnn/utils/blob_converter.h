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
#include "tnn/core/macro.h"
#include "tnn/core/status.h"

#pragma warning(push)
#pragma warning(disable : 4251)

namespace TNN_NS {

typedef enum {
    INVALID    = -1,
    N8UC3      = 0x00,
    N8UC4      = 0x01,
    NGRAY      = 0x10,
    NNV21      = 0x11,
    NNV12      = 0x12,
    NCHW_FLOAT = 0x20,
    // RESERVED FOR INTERNAL TEST USE
    RESERVED_BFP16_TEST = 0x200,
    RESERVED_FP16_TEST  = 0x201,
    RESERVED_INT8_TEST  = 0x202,
} PUBLIC MatType;

class PUBLIC Mat {
public:
    ~Mat();
    
    Mat(DeviceType device_type, MatType mat_type, DimsVector shape_dims, void* data);
    Mat(DeviceType device_type, MatType mat_type, DimsVector shape_dims);

    DEPRECATED("use Mat(DeviceType, MatType, DimsVector, void*) instead")
    Mat(DeviceType device_type, MatType mat_type, void* data) : Mat(device_type, mat_type, {1,0,0,0}, data) {};
    
public:
    DeviceType GetDeviceType();
    MatType GetMatType();
    void* GetData();
    int GetBatch();
    int GetChannel();
    int GetHeight();
    int GetWidth();
    int GetDim(int index);
    DimsVector GetDims();

    //    Mat(const Mat&) = delete;
    //    Mat& operator=(const Mat&) = delete;

private:
    Mat(){};

protected:
    TNN_NS::DeviceType device_type_ = DEVICE_NAIVE;
    TNN_NS::MatType mat_type_       = INVALID;
    void* data_                     = nullptr;
    DimsVector dims_ = {};

private:
    std::shared_ptr<void> data_alloc_ = nullptr;
};

using MatMap = std::map<std::string, std::shared_ptr<Mat>>;

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
    virtual Status ConvertToMat(Mat& image, MatConvertParam param, void* command_queue);
    virtual Status ConvertFromMat(Mat& image, MatConvertParam param, void* command_queue);

    virtual Status ConvertToMatAsync(Mat& image, MatConvertParam param, void* command_queue);
    virtual Status ConvertFromMatAsync(Mat& image, MatConvertParam param, void* command_queue);

private:
    Blob* blob_ = nullptr;
    std::shared_ptr<BlobConverterAcc> impl_ = nullptr;
};

}  // namespace TNN_NS

#pragma warning(pop)

#endif  // TNN_INCLUDE_TNN_UTILS_BLOB_CONVERTER_H_
