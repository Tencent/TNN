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

#include "tnn/memory_manager/share_one_thread_memory_mode_state.h"

namespace TNN_NS {

ShareOneThreadMemoryModeState::ShareOneThreadMemoryModeState() {
    init_thread_id_ = std::this_thread::get_id();
}

Status ShareOneThreadMemoryModeState::GetStatus() {
    std::thread::id current_thread_id = std::this_thread::get_id();
    if (memory_allocated && (current_thread_id == init_thread_id_)) {
        return TNN_OK;
    } else if (!memory_allocated) {
        return Status(TNNERR_FORWARD_MEM_NOT_SET, "memory is not set");
    } else {
        return Status(TNNERR_SHARED_MEMORY_FORWARD_NOT_SAME_THREAD, "memory canbe shared only in the same thread");
    }
}

}  // namespace TNN_NS
