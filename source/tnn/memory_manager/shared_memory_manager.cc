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

#include "tnn/memory_manager/shared_memory_manager.h"

namespace TNN_NS {

bool operator<(SharedMemoryId lhs, SharedMemoryId rhs) {
    if ((lhs.thread_id < rhs.thread_id) || ((lhs.thread_id == rhs.thread_id) && lhs.device_type < rhs.device_type) ||
        ((lhs.thread_id == rhs.thread_id) && (lhs.device_type == rhs.device_type) && (lhs.device_id < rhs.device_id))) {
        return true;
    } else {
        return false;
    }
}

std::map<SharedMemoryId, SharedMemory> SharedMemoryManager::s_shared_forward_memory;
std::map<SharedMemoryId, std::vector<ISharedMemoryChangeListener *>> SharedMemoryManager::s_shared_memory_instances;

SharedMemory SharedMemoryManager::GetSharedMemory(int forward_memory_size, std::thread::id thread_id,
                                                  AbstractDevice *device, int device_id,
                                                  ISharedMemoryChangeListener *listener,
                                                  Status &status) {
    SharedMemoryId memory_id;
    memory_id.thread_id                                          = thread_id;
    memory_id.device_type                                        = device->GetDeviceType();
    memory_id.device_id                                          = device_id;
    SharedMemory &share_memory                                   = s_shared_forward_memory[memory_id];
    std::vector<ISharedMemoryChangeListener *> &shared_instances = s_shared_memory_instances[memory_id];
    if (forward_memory_size > share_memory.shared_memory_size) {
        void *new_shared_memory = NULL;
        BlobMemorySizeInfo info;
        info.data_type = DATA_TYPE_INT8; 
        info.dims.push_back(forward_memory_size);
        status = device->Allocate(&new_shared_memory, info);
        if (status != TNN_OK) {
            return SharedMemory();
        }

        if (share_memory.shared_memory_data != NULL) {
            device->Free(share_memory.shared_memory_data);
        }
 
        for (int i = 0; i < shared_instances.size(); ++i) {
            shared_instances[i]->OnSharedForwardMemoryChanged(new_shared_memory);
        }
        share_memory.shared_memory_data = new_shared_memory;
        share_memory.shared_memory_size = forward_memory_size;
    }
    share_memory.shared_memory_ref_count++;
    shared_instances.push_back(listener);
    return share_memory;
}

void SharedMemoryManager::ReleaseSharedMemory(std::thread::id thread_id, AbstractDevice *device, int device_id,
                                              ISharedMemoryChangeListener *listener) {
    SharedMemoryId memory_id;
    memory_id.thread_id                                          = thread_id;
    memory_id.device_type                                        = device->GetDeviceType();
    memory_id.device_id                                          = device_id;
    std::vector<ISharedMemoryChangeListener *> &shared_instances = s_shared_memory_instances[memory_id];
    std::vector<ISharedMemoryChangeListener *>::iterator it =
        std::find(shared_instances.begin(), shared_instances.end(), listener);
    if (it != shared_instances.end()) {
        shared_instances.erase(it);
    }
    SharedMemory &memory = s_shared_forward_memory[memory_id];
    memory.shared_memory_ref_count--;
    if (memory.shared_memory_ref_count == 0) {
        device->Free(memory.shared_memory_data);
        s_shared_forward_memory.erase(memory_id);
    }
}

}  // namespace TNN_NS
