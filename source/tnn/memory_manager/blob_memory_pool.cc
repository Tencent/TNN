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

#include "tnn/memory_manager/blob_memory_pool.h"

#include <limits.h>

#include <map>
#include <tuple>

namespace TNN_NS {

BlobMemoryPool::BlobMemoryPool(AbstractDevice *device) {
    all_blob_memory_size_ = 0;
    device_               = device;
}

BlobMemoryPool::~BlobMemoryPool() {
    ClearBlobMemoryPool();
}

void BlobMemoryPool::ClearBlobMemoryPool() {
    auto blob_memory_library = blob_memory_library_;
    for (auto &iter : blob_memory_library) {
        delete iter;
    }
    
    blob_memory_library_.clear();
    all_blob_memory_size_ = 0;
}

AbstractDevice *BlobMemoryPool::GetDevice() {
    return device_;
}

BlobMemory *BlobMemoryPool::BorrowBlobMemory(int use_count, BlobMemorySizeInfo &size_info, bool use_new_memory) {
    if (use_new_memory) {
        BlobMemory *blob_memory = CreateBlobMemory(use_count, size_info);
        blob_memory_library_.insert(blob_memory);
        return blob_memory;
    } else {
        BlobMemoryNode *blob_node = ExtractNearestBlobMemoryNode(size_info);
        if (blob_node == NULL) {
            BlobMemory *blob_memory = CreateBlobMemory(use_count, size_info);
            blob_memory_library_.insert(blob_memory);
            return blob_memory;
        } else {
            BlobMemory *blob_memory = blob_node->blob_memory;
            blob_memory->UpdateBlobMemorySizeInfo(size_info);
            blob_memory->SetUseCount(use_count);
            delete blob_node;
            return blob_memory;
        }
    }
}

void BlobMemoryPool::RefundBlobMemory(BlobMemory *blob_memory) {
    ASSERT(blob_memory != NULL);
    DataType data_type          = blob_memory->GetBlobMemorySizeInfo().data_type;
    BlobMemoryNode *list_header = GetBlobMemoryNodeListHeader(data_type);
    BlobMemoryNode *new_header  = new BlobMemoryNode();
    new_header->blob_memory     = blob_memory;
    new_header->next            = list_header;
    SetBlobMemoryNodeListHeader(data_type, new_header);
}

void BlobMemoryPool::ReleaseBlobMemoryNodeList(BlobMemoryNode *list_header) {
    while (list_header) {
        auto temp   = list_header;
        list_header = list_header->next;
        delete temp;
    }
}

BlobMemoryNode *BlobMemoryPool::ExtractNearestBlobMemoryNode(BlobMemorySizeInfo &size_info) {
    BlobMemoryNode *list_header = GetBlobMemoryNodeListHeader(size_info.data_type);
    if (!list_header) {
        return nullptr;
    }

    BlobMemoryNode *node_prev                                         = nullptr;
    BlobMemoryNode *node_cur                                          = list_header;
    std::tuple<BlobMemoryNode *, BlobMemoryNode *, int64_t> min_diff_area = std::make_tuple(nullptr, nullptr, LLONG_MAX);
    while (node_cur) {
        int64_t bytes_diff = ResolveBlobMemoryNodeBytesDiff(size_info, node_cur);

        if (bytes_diff < std::get<2>(min_diff_area)) {
            min_diff_area = std::make_tuple(node_prev, node_cur, bytes_diff);
        }

        node_prev = node_cur;
        node_cur  = node_cur->next;
    }

    node_prev = std::get<0>(min_diff_area);
    node_cur  = std::get<1>(min_diff_area);
    ASSERT(node_cur != nullptr);

    if (node_prev) {
        node_prev->next = node_cur->next;
    } else {
        list_header = node_cur->next;
        SetBlobMemoryNodeListHeader(size_info.data_type, list_header);
    }
    return node_cur;
}

Status BlobMemoryPool::AssignAllBlobMemory(MemoryAssignStrategy &strategy) {
    return strategy.AssignAllBlobMemory(blob_memory_library_);
}

int BlobMemoryPool::GetAllBlobMemorySize() {
    CalculateAllBlobMemorySize();
    return all_blob_memory_size_;
}

void BlobMemoryPool::CalculateAllBlobMemorySize() {
    typename std::set<BlobMemory *>::iterator iter;
    all_blob_memory_size_ = 0;
    for (auto iter : blob_memory_library_) {
        BlobMemorySizeInfo info = iter->GetBlobMemorySizeInfo();
        all_blob_memory_size_ += GetBlobMemoryBytesSize(info);
    }
}

}  // namespace TNN_NS
