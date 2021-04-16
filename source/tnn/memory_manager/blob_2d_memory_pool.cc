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

#include "tnn/memory_manager/blob_2d_memory_pool.h"
#include "tnn/memory_manager/blob_2d_memory.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

Blob2DMemoryPool::Blob2DMemoryPool(AbstractDevice *device) : BlobMemoryPool(device) {
    blob_memory_list_header_map_.clear();
}

Blob2DMemoryPool::~Blob2DMemoryPool() {
    ClearBlobMemoryPool();
}

void Blob2DMemoryPool::ClearBlobMemoryPool() {
    BlobMemoryPool::ClearBlobMemoryPool();
    
    auto blob_memory_list_header_map = blob_memory_list_header_map_;
    for (auto iter : blob_memory_list_header_map) {
        auto list_header = iter.second;
        ReleaseBlobMemoryNodeList(list_header);
    }
    blob_memory_list_header_map_.clear();
}

BlobMemory *Blob2DMemoryPool::CreateBlobMemory(int use_count, BlobMemorySizeInfo &size_info) {
    return new Blob2DMemory(device_, size_info, use_count);
}

BlobMemoryNode *Blob2DMemoryPool::GetBlobMemoryNodeListHeader(DataType data_type) {
    return blob_memory_list_header_map_[data_type];
}

void Blob2DMemoryPool::SetBlobMemoryNodeListHeader(DataType data_type, BlobMemoryNode *new_header) {
    blob_memory_list_header_map_[data_type] = new_header;
}

int64_t Blob2DMemoryPool::ResolveBlobMemoryNodeBytesDiff(BlobMemorySizeInfo &size_info, BlobMemoryNode *node) {
    int64_t target_bytes_size = GetBlobMemoryBytesSize(size_info);

    auto node_cur_info       = node->blob_memory->GetBlobMemorySizeInfo();
    int64_t node_cur_bytes_size = GetBlobMemoryBytesSize(node_cur_info);

    BlobMemorySizeInfo max_info;
    max_info.data_type = size_info.data_type;
    max_info.dims      = DimsVectorUtils::Max(size_info.dims, node_cur_info.dims);
    int64_t max_bytes_size = GetBlobMemoryBytesSize(max_info);


    if (size_info.dims[0] <= node_cur_info.dims[0] && size_info.dims[1] <= node_cur_info.dims[1]) {
        return max_bytes_size - target_bytes_size;
    } else {
        return max_bytes_size - node_cur_bytes_size;
    }
}

BlobMemoryNode *Blob2DMemoryPool::ExtractNearestBlobMemoryNode(BlobMemorySizeInfo &size_info) {
    BlobMemoryNode *list_header = GetBlobMemoryNodeListHeader(size_info.data_type);
    if (!list_header) {
        return nullptr;
    }

    BlobMemoryNode *node_prev                                           = nullptr;
    BlobMemoryNode *node_cur                                            = list_header;
    std::tuple<BlobMemoryNode *, BlobMemoryNode *, int64_t> min_diff_exist  = std::make_tuple(nullptr, nullptr, LLONG_MAX);
    std::tuple<BlobMemoryNode *, BlobMemoryNode *, int64_t> min_diff_extend = std::make_tuple(nullptr, nullptr, LLONG_MAX);
    while (node_cur) {
        int64_t bytes_diff = ResolveBlobMemoryNodeBytesDiff(size_info, node_cur);

        auto node_cur_sizeinfo = node_cur->blob_memory->GetBlobMemorySizeInfo();
        ASSERT(2 == size_info.dims.size() && 2 == node_cur_sizeinfo.dims.size());
        if (size_info.dims[0] <= node_cur_sizeinfo.dims[0] && size_info.dims[1] <= node_cur_sizeinfo.dims[1]) {
            // the memory pool have the blob to fit the size_info
            if (bytes_diff < std::get<2>(min_diff_exist)) {
                min_diff_exist = std::make_tuple(node_prev, node_cur, bytes_diff);
            }
        } else {
            int target_bytes_size = GetBlobMemoryBytesSize(size_info);
            if (bytes_diff < target_bytes_size) {
                // can extend
                if (bytes_diff < std::get<2>(min_diff_extend)) {
                    min_diff_extend = std::make_tuple(node_prev, node_cur, bytes_diff);
                }
            }
        }

        node_prev = node_cur;
        node_cur  = node_cur->next;
    }

    if (nullptr != std::get<1>(min_diff_exist)) {
        node_prev = std::get<0>(min_diff_exist);
        node_cur  = std::get<1>(min_diff_exist);
    } else if (nullptr != std::get<1>(min_diff_extend)) {
        node_prev = std::get<0>(min_diff_extend);
        node_cur  = std::get<1>(min_diff_extend);
    } else {
        return nullptr;
    }

    ASSERT(node_cur != nullptr);

    if (node_prev) {
        node_prev->next = node_cur->next;
    } else {
        list_header = node_cur->next;
        SetBlobMemoryNodeListHeader(size_info.data_type, list_header);
    }
    return node_cur;
}

}  // namespace TNN_NS
