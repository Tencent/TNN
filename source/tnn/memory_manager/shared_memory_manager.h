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

#ifndef TNN_SOURCE_TNN_MEMORY_MANAGER_SHARED_MEMORY_MANAGER_H_
#define TNN_SOURCE_TNN_MEMORY_MANAGER_SHARED_MEMORY_MANAGER_H_

#include <algorithm>
#include <map>
#include <queue>
#include <set>
#include <thread>
#include <vector>
#include "tnn/core/abstract_device.h"
#include "tnn/core/common.h"

namespace TNN_NS {

struct SharedMemory {
    int shared_memory_size      = 0;
    void *shared_memory_data    = NULL;
    int shared_memory_ref_count = 0;
};

struct SharedMemoryId {
    std::thread::id thread_id;
    DeviceType device_type;
    int device_id;
};

class ISharedMemoryChangeListener {
public:
    virtual void OnSharedForwardMemoryChanged(void *memory) = 0;
};

bool operator<(SharedMemoryId lhs, SharedMemoryId rhs);

class SharedMemoryManager {
public:
    static SharedMemory GetSharedMemory(
        int forward_memory_size, std::thread::id thread_id,
        AbstractDevice *device, int device_id,
        ISharedMemoryChangeListener *listener,
        Status &status);

    static void ReleaseSharedMemory(std::thread::id thread_id,
                                    AbstractDevice *device, int device_id,
                                    ISharedMemoryChangeListener *listener);

private:
    static std::map<SharedMemoryId, SharedMemory> s_shared_forward_memory;
    static std::map<SharedMemoryId, std::vector<ISharedMemoryChangeListener *>>
        s_shared_memory_instances;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_MEMORY_MANAGER_SHARED_MEMORY_MANAGEER_H_
