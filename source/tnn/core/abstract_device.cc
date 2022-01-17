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

#include "tnn/core/abstract_device.h"

#include <map>
#include <mutex>

namespace TNN_NS {

AbstractDevice::AbstractDevice(DeviceType device_type) : device_type_(device_type) {}

AbstractDevice::~AbstractDevice() {}

DeviceType AbstractDevice::GetDeviceType() {
    return device_type_;
}

Status AbstractDevice::Allocate(BlobHandle* handle, BlobMemorySizeInfo& size_info) {
    void* data = nullptr;

    auto status = Allocate(&data, size_info);
    if (status != TNN_OK) {
        return status;
    }
    handle->base         = data;
    handle->bytes_offset = 0;

    return TNN_OK;
}

std::shared_ptr<const ImplementedPrecision> AbstractDevice::GetImplementedPrecision(LayerType type) {
    return std::make_shared<ImplementedPrecision>();
}

std::shared_ptr<const ImplementedLayout> AbstractDevice::GetImplementedLayout(LayerType type) {
    return std::make_shared<ImplementedLayout>();
}

AbstractDevice* GetDevice(DeviceType type) {
    return GetGlobalDeviceMap()[type].get();
}

/*
 * All devices are stored in this map.
 * The actual Device is registered as runtime.
 */
std::map<DeviceType, std::shared_ptr<AbstractDevice>>& GetGlobalDeviceMap() {
    static std::once_flag once;
    static std::shared_ptr<std::map<DeviceType, std::shared_ptr<AbstractDevice>>> device_map;
    std::call_once(once, []() { device_map.reset(new std::map<DeviceType, std::shared_ptr<AbstractDevice>>); });
    return *device_map;
}

}  // namespace TNN_NS

