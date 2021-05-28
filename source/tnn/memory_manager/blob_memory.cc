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

#include "tnn/memory_manager/blob_memory.h"

namespace TNN_NS {

BlobMemory::BlobMemory(AbstractDevice* device, BlobMemorySizeInfo& size_info, int use_count)
    : device_(device), size_info_(size_info), use_count_(use_count) {
    need_release_memory_ = false;
}
BlobMemory::~BlobMemory() {
    if (need_release_memory_) {
        need_release_memory_ = false;
        device_->Free(handle_.base);
    }
}

BlobMemorySizeInfo BlobMemory::GetBlobMemorySizeInfo() const {
    return size_info_;
}

void BlobMemory::SetUseCount(int use_count) {
    use_count_ = use_count;
}

int BlobMemory::GetUseCount() const {
    return use_count_;
}

bool BlobMemory::DecrementUseCount() {
    if (use_count_ > 0) {
        --use_count_;
        return true;
    } else {
        return false;
    }
}

Status BlobMemory::AllocateHandle() {
    auto status = device_->Allocate(&handle_, size_info_);
    if (status != TNN_OK) {
        return status;
    }

    need_release_memory_ = true;
    return TNN_OK;
}

void BlobMemory::SetHandleFromExternal(BlobHandle handle) {
    handle_              = handle;
    need_release_memory_ = false;
}

BlobHandle BlobMemory::GetHandle() {
    return handle_;
}

}  // namespace TNN_NS
