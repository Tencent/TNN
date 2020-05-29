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

#ifndef TNN_SOURCE_TNN_UTILS_BLOB_CONVERTER_INTERNAL_H_
#define TNN_SOURCE_TNN_UTILS_BLOB_CONVERTER_INTERNAL_H_

#include <map>
#include <memory>

#include "tnn/core/blob.h"
#include "tnn/core/common.h"
#include "tnn/utils/blob_converter.h"

namespace TNN_NS {

class BlobConverterAcc {
public:
    BlobConverterAcc(Blob* blob) : blob_(blob){};
    virtual ~BlobConverterAcc(){};
    virtual Status ConvertToMat(Mat& image, MatConvertParam param, void* command_queue = NULL)      = 0;
    virtual Status ConvertToMatAsync(Mat& image, MatConvertParam param, void* command_queue = NULL) = 0;

    virtual Status ConvertFromMat(Mat& image, MatConvertParam param, void* command_queue = NULL)      = 0;
    virtual Status ConvertFromMatAsync(Mat& image, MatConvertParam param, void* command_queue = NULL) = 0;

protected:
    Blob* blob_;
};

class BlobConverterAccCreater {
public:
    virtual ~BlobConverterAccCreater(){};
    virtual std::shared_ptr<BlobConverterAcc> CreateBlobConverterAcc(Blob* blob) = 0;
};

class BlobConverterManager {
public:
    static std::shared_ptr<BlobConverterManager>& Shared();
    BlobConverterManager();
    ~BlobConverterManager();
    std::shared_ptr<BlobConverterAcc> CreateBlobConverterAcc(Blob* blob);
    int RegisterBlobConverterAccCreater(DeviceType type, std::shared_ptr<BlobConverterAccCreater> creater);

private:
    std::map<DeviceType, std::shared_ptr<BlobConverterAccCreater>> converter_creater_map_;
};

template <typename T>
class BlobConverterAccRegister {
public:
    explicit BlobConverterAccRegister(DeviceType type) {
        auto creater  = std::make_shared<T>();
        auto& manager = BlobConverterManager::Shared();
        manager->RegisterBlobConverterAccCreater(type, creater);
    };
    ~BlobConverterAccRegister(){};
};
}  // namespace TNN_NS

#define DECLARE_BLOB_CONVERTER_CREATER(device)                                                                         \
    class device##BlobConverterAccCreater : public BlobConverterAccCreater {                                           \
    public:                                                                                                            \
        virtual ~device##BlobConverterAccCreater(){};                                                                  \
        virtual std::shared_ptr<BlobConverterAcc> CreateBlobConverterAcc(Blob* blob) {                                 \
            return std::make_shared<device##BlobConverterAcc>(blob);                                                   \
        };                                                                                                             \
    }

#define REGISTER_BLOB_CONVERTER(device, device_type)                                                                   \
    BlobConverterAccRegister<device##BlobConverterAccCreater> g_blob_converter_##device(device_type)

#endif  // TNN_SOURCE_TNN_UTILS_BLOB_CONVERTER_INTERNAL_H_
