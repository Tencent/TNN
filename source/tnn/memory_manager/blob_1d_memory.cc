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

#include "tnn/memory_manager/blob_1d_memory.h"

namespace TNN_NS {

Blob1DMemory::Blob1DMemory(AbstractDevice* device, BlobMemorySizeInfo& size_info, int use_count)
    : BlobMemory(device, size_info, use_count) {}

Blob1DMemory::~Blob1DMemory() {}

void Blob1DMemory::UpdateBlobMemorySizeInfo(BlobMemorySizeInfo info) {
    int current_bytes_size = GetBlobMemoryBytesSize(size_info_);
    int new_bytes_size     = GetBlobMemoryBytesSize(info);
    if (new_bytes_size > current_bytes_size) {
        size_info_ = info;
    }
}

}  // namespace TNN_NS
