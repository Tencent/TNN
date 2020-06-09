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

#include <mutex>
#include <string>
#include "tnn/utils/blob_converter.h"

#include "tnn/utils/blob_converter_internal.h"

namespace TNN_NS {

BlobConverter::BlobConverter(Blob* blob) {
    blob_ = blob;
    impl_ = BlobConverterManager::Shared()->CreateBlobConverterAcc(blob);
}

Status BlobConverter::ConvertToMat(Mat& image, MatConvertParam param, void* command_queue) {
    if (!impl_) {
        return Status(TNNERR_INIT_LAYER, "image converter is nil, check device type");
    }

    return impl_->ConvertToMat(image, param, command_queue);
}

Status BlobConverter::ConvertToMatAsync(Mat& image, MatConvertParam param, void* command_queue) {
    if (!impl_) {
        return Status(TNNERR_INIT_LAYER, "image converter is nil, check device type");
    }

    return impl_->ConvertToMatAsync(image, param, command_queue);
}

Status BlobConverter::ConvertFromMat(Mat& image, MatConvertParam param, void* command_queue) {
    if (!impl_) {
        return Status(TNNERR_INIT_LAYER, "image converter is nil, check device type");
    }

    return impl_->ConvertFromMat(image, param, command_queue);
}

Status BlobConverter::ConvertFromMatAsync(Mat& image, MatConvertParam param, void* command_queue) {
    if (!impl_) {
        return Status(TNNERR_INIT_LAYER, "image converter is nil, check device type");
    }

    return impl_->ConvertFromMatAsync(image, param, command_queue);
}

std::shared_ptr<BlobConverterManager>& BlobConverterManager::Shared() {
    static std::once_flag once;
    static std::shared_ptr<BlobConverterManager> g_global_blob_converter_manager;
    std::call_once(once, []() { g_global_blob_converter_manager = std::make_shared<BlobConverterManager>(); });
    return g_global_blob_converter_manager;
}

std::shared_ptr<BlobConverterAcc> BlobConverterManager::CreateBlobConverterAcc(Blob* blob) {
    auto iter = converter_creater_map_.find(blob->GetBlobDesc().device_type);
    if (iter != converter_creater_map_.end()) {
        return iter->second->CreateBlobConverterAcc(blob);
    }
    return nullptr;
}

int BlobConverterManager::RegisterBlobConverterAccCreater(DeviceType type,
                                                          std::shared_ptr<BlobConverterAccCreater> creater) {
    auto iter = converter_creater_map_.find(type);
    if (iter != converter_creater_map_.end()) {
        LOGE("Error: device_type(%d) cannot be registered twice\n", type);
        return 1;
    }
    if (!creater) {
        LOGE("Error: MatBlobConverterAccCreater is nil device_type(%d)\n", type);
        return 1;
    }
    converter_creater_map_[type] = creater;
    return 0;
}

BlobConverterManager::BlobConverterManager() {}
BlobConverterManager::~BlobConverterManager() {}

}  // namespace TNN_NS
