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

#include "tnn/memory_manager/blob_1d_memory_pool.h"
#include <cmath>
#include "tnn/memory_manager/blob_1d_memory.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

Blob1DMemoryPool::Blob1DMemoryPool(AbstractDevice* device) : BlobMemoryPool(device) {
    blob_memory_list_header_ = NULL;
}

Blob1DMemoryPool::~Blob1DMemoryPool() {
    ClearBlobMemoryPool();
}

void Blob1DMemoryPool::ClearBlobMemoryPool() {
    BlobMemoryPool::ClearBlobMemoryPool();
    
    ReleaseBlobMemoryNodeList(blob_memory_list_header_);
    blob_memory_list_header_ = NULL;
}

BlobMemory* Blob1DMemoryPool::CreateBlobMemory(int use_count, BlobMemorySizeInfo& size_info) {
    return new Blob1DMemory(device_, size_info, use_count);
}

BlobMemoryNode* Blob1DMemoryPool::GetBlobMemoryNodeListHeader(DataType data_type) {
    return blob_memory_list_header_;
}

void Blob1DMemoryPool::SetBlobMemoryNodeListHeader(DataType data_type, BlobMemoryNode* new_header) {
    blob_memory_list_header_ = new_header;
}

int64_t Blob1DMemoryPool::ResolveBlobMemoryNodeBytesDiff(BlobMemorySizeInfo& size_info, BlobMemoryNode* node) {
    int64_t target_bytes_size = GetBlobMemoryBytesSize(size_info);
    auto node_cur_info    = node->blob_memory->GetBlobMemorySizeInfo();
    int64_t node_bytes_size   = GetBlobMemoryBytesSize(node_cur_info);
    return std::abs(target_bytes_size - node_bytes_size);
}

}  // namespace TNN_NS
