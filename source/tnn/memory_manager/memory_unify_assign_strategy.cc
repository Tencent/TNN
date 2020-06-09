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

#include "tnn/memory_manager/memory_unify_assign_strategy.h"
#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {

MemoryUnifyAssignStrategy::MemoryUnifyAssignStrategy(void* data) {
    all_blob_memory_data_ = data;
}

Status MemoryUnifyAssignStrategy::AssignAllBlobMemory(std::set<BlobMemory*>& blob_memory_library) {
    int blob_memory_start_offset = 0;
    for (auto& iter : blob_memory_library) {
        BlobHandle handle;
        handle.base         = all_blob_memory_data_;
        handle.bytes_offset = blob_memory_start_offset;
        iter->SetHandleFromExternal(handle);
        BlobMemorySizeInfo size_info = iter->GetBlobMemorySizeInfo();
        blob_memory_start_offset += GetBlobMemoryBytesSize(size_info);
    }
    return TNN_OK;
}

}  // namespace TNN_NS

// namespace TNN_NS
