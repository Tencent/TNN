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

#include "tnn/utils/mat_converter.h"
#include "tnn/utils/mat_converter_internal.h"

namespace TNN_NS {

MatConverter::MatConverter(Mat* src, Mat* dst) {
    DeviceType device_type = DEVICE_NAIVE;
    // get device type
    DeviceType src_dt = src->GetDeviceType();
    DeviceType dst_dt = dst->GetDeviceType();
    if (src_dt == dst_dt) {
        device_type = src_dt;
    } else if (DEVICE_NAIVE == src_dt || DEVICE_ARM == src_dt) {
        device_type = dst_dt;
    } else if (DEVICE_NAIVE == dst_dt || DEVICE_ARM == dst_dt) {
        device_type = src_dt;
    } else {
        impl_ = nullptr;
        return;
    }

    impl_ = MatConverterManager::Shared()->CreateMatConverterAcc(device_type);
}

Status MatConverter::Resize(Mat& src, Mat& dst, ResizeParam param, void* command_queue) {
    if (!impl_) {
        return Status(TNNERR_NULL_PARAM, "mat converter is nil, check device type");
    }

    return impl_->Resize(src, dst, param, command_queue);
}

Status MatConverter::ResizeAndPaste(Mat& src, Mat& dst, ResizeParam param, PasteParam paste_param, void* command_queue) {
    if (!impl_) {
        return Status(TNNERR_NULL_PARAM, "mat converter is nil, check device type");
    }

    return impl_->ResizeAndPaste(src, dst, param, paste_param, command_queue);
}

Status MatConverter::Crop(Mat& src, Mat& dst, CropParam param, void* command_queue) {
    if (!impl_) {
        return Status(TNNERR_NULL_PARAM, "mat converter is nil, check device type");
    }

    return impl_->Crop(src, dst, param, command_queue);
}

Status MatConverter::WarpAffine(Mat& src, Mat& dst, WarpAffineParam param, void* command_queue) {
    if (!impl_) {
        return Status(TNNERR_NULL_PARAM, "mat converter is nil, check device type");
    }

    return impl_->WarpAffine(src, dst, param, command_queue);
}

Status MatConverter::ConcatMatWithBatch(std::vector<Mat>& src_vec, Mat& dst, void* command_queue) {
    if (!impl_) {
        return Status(TNNERR_NULL_PARAM, "mat converter is nil, check device type");
    }

    return impl_->ConcatMatWithBatch(src_vec, dst, command_queue);
}

std::shared_ptr<MatConverterManager>& MatConverterManager::Shared() {
    static std::once_flag once;
    static std::shared_ptr<MatConverterManager> g_global_blob_converter_manager;
    std::call_once(once, []() { g_global_blob_converter_manager = std::make_shared<MatConverterManager>(); });
    return g_global_blob_converter_manager;
}

std::shared_ptr<MatConverterAcc> MatConverterManager::CreateMatConverterAcc(DeviceType device_type) {
    auto iter = converter_creater_map_.find(device_type);
    if (iter != converter_creater_map_.end()) {
        return iter->second->CreateMatConverterAcc();
    }
    return nullptr;
}

int MatConverterManager::RegisterMatConverterAccCreater(DeviceType type,
                                                        std::shared_ptr<MatConverterAccCreater> creater) {
    auto iter = converter_creater_map_.find(type);
    if (iter != converter_creater_map_.end()) {
        LOGE("Error: device_type(%d) cannot be registered twice\n", type);
        return 1;
    }
    if (!creater) {
        LOGE("Error: MatConverterAccCreater is nil device_type(%d)\n", type);
        return 1;
    }
    converter_creater_map_[type] = creater;
    return 0;
}

MatConverterManager::MatConverterManager() {}
MatConverterManager::~MatConverterManager() {}

}  // namespace TNN_NS
