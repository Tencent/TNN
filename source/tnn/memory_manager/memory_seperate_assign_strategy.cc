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

#include "tnn/memory_manager/memory_seperate_assign_strategy.h"
#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {

Status MemorySeperateAssignStrategy::AssignAllBlobMemory(std::set<BlobMemory*>& blob_memory_library) {
    typename std::set<BlobMemory*>::iterator iter;
    for (auto iter : blob_memory_library) {
        auto status = iter->AllocateHandle();
        if (status != TNN_OK) {
            return status;
        }
    }
    return TNN_OK;
}

}  // namespace TNN_NS
