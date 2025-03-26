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

#include "tnn/utils/device_utils.h"

#include <map>
#include <mutex>

#include "tnn/core/macro.h"
#include "tnn/core/common.h"
#include "tnn/core/status.h"
#include "tnn/core/abstract_device.h"

namespace TNN_NS {

// @brief Get a specified device from a device group
DeviceType GetConcreteDeviceType(DeviceType type) {
    std::mutex g_map_mutex;
    std::lock_guard<std::mutex> guard(g_map_mutex);

    switch(type) {
        case DEVICE_GROUP_CPU:
            {
                if (GetDevice(DEVICE_X86) != nullptr) {
                    return DEVICE_X86;
                } else if (GetDevice(DEVICE_ARM) != nullptr) {
                    return DEVICE_ARM;
                } else if (GetDevice(DEVICE_NAIVE) != nullptr) {
                    return DEVICE_NAIVE;
                }
                break;
            }
        case DEVICE_GROUP_GPU:
            {
                if (GetDevice(DEVICE_CUDA) != nullptr) {
                    return DEVICE_CUDA;
                } else if (GetDevice(DEVICE_OPENCL) != nullptr) {
                    return DEVICE_OPENCL;
                } else if (GetDevice(DEVICE_METAL) != nullptr) {
                    return DEVICE_METAL;
                }
                break;
            }
        case DEVICE_GROUP_NPU:
            {
                if (GetDevice(DEVICE_DSP) != nullptr) {
                    return DEVICE_DSP;
                } else if (GetDevice(DEVICE_ATLAS) != nullptr) {
                    return DEVICE_ATLAS;
                } else if (GetDevice(DEVICE_HUAWEI_NPU) != nullptr) {
                    return DEVICE_HUAWEI_NPU;
                } else if (GetDevice(DEVICE_RK_NPU) != nullptr) {
                    return DEVICE_RK_NPU;
                }
                break;
            }
        default:
            break;
    }

    LOGI("No concrete device available.\n");
    return DEVICE_NAIVE;
}

}  // namespace TNN_NS
