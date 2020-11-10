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

#ifndef TNN_SOURCE_TNN_UTILS_MAT_CONVERTER_INTERNAL_H_
#define TNN_SOURCE_TNN_UTILS_MAT_CONVERTER_INTERNAL_H_

#include <map>
#include <memory>

#include "tnn/core/blob.h"
#include "tnn/core/common.h"
#include "tnn/utils/blob_converter.h"
#include "tnn/utils/mat_utils.h"

namespace TNN_NS {

class MatConverterAcc {
public:
    MatConverterAcc(){};
    virtual ~MatConverterAcc(){};
    virtual Status Copy(Mat& src, Mat& dst, void* command_queue = NULL)                                      = 0;
    virtual Status Resize(Mat& src, Mat& dst, ResizeParam param, void* command_queue = NULL)                 = 0;
    virtual Status Crop(Mat& src, Mat& dst, CropParam param, void* command_queue = NULL)                     = 0;
    virtual Status WarpAffine(Mat& src, Mat& dst, WarpAffineParam param, void* command_queue = NULL)         = 0;
    virtual Status CvtColor(Mat& src, Mat& dst, ColorConversionType type, void* command_queue = NULL)        = 0;
    virtual Status CopyMakeBorder(Mat& src, Mat& dst, CopyMakeBorderParam param, void* command_queue = NULL) = 0;
};

class MatConverterAccCreater {
public:
    virtual ~MatConverterAccCreater(){};
    virtual std::shared_ptr<MatConverterAcc> CreateMatConverterAcc() = 0;
};

class MatConverterManager {
public:
    static std::shared_ptr<MatConverterManager>& Shared();
    MatConverterManager();
    ~MatConverterManager();
    std::shared_ptr<MatConverterAcc> CreateMatConverterAcc(DeviceType device_type);
    int RegisterMatConverterAccCreater(DeviceType type, std::shared_ptr<MatConverterAccCreater> creater);

private:
    std::map<DeviceType, std::shared_ptr<MatConverterAccCreater>> converter_creater_map_;
};

template <typename T>
class MatConverterAccRegister {
public:
    explicit MatConverterAccRegister(DeviceType type) {
        auto creater  = std::make_shared<T>();
        auto& manager = MatConverterManager::Shared();
        manager->RegisterMatConverterAccCreater(type, creater);
    };
    ~MatConverterAccRegister(){};
};
}  // namespace TNN_NS

#define DECLARE_MAT_CONVERTER_CREATER(device)                                                                          \
    class device##MatConverterAccCreater : public MatConverterAccCreater {                                             \
    public:                                                                                                            \
        virtual ~device##MatConverterAccCreater(){};                                                                   \
        virtual std::shared_ptr<MatConverterAcc> CreateMatConverterAcc() {                                             \
            return std::make_shared<device##MatConverterAcc>();                                                        \
        };                                                                                                             \
    }

#define REGISTER_MAT_CONVERTER(device, device_type)                                                                    \
    MatConverterAccRegister<device##MatConverterAccCreater> g_mat_converter_##device(device_type)

#endif  // TNN_SOURCE_TNN_UTILS_MAT_CONVERTER_INTERNAL_H_
