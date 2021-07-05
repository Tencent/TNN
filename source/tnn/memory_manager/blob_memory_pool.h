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

#ifndef TNN_SOURCE_TNN_MEMORY_MANAGER_BLOB_MEMORY_POOL_H_
#define TNN_SOURCE_TNN_MEMORY_MANAGER_BLOB_MEMORY_POOL_H_

#include <queue>
#include <set>

#include "tnn/core/abstract_device.h"
#include "tnn/memory_manager/blob_memory.h"
#include "tnn/memory_manager/memory_seperate_assign_strategy.h"
#include "tnn/memory_manager/memory_unify_assign_strategy.h"

namespace TNN_NS {
struct BlobMemoryNode {
    BlobMemory *blob_memory = nullptr;
    BlobMemoryNode *next    = nullptr;
};

class BlobMemoryPool {
public:
    explicit BlobMemoryPool(AbstractDevice *device);
    virtual ~BlobMemoryPool();
    BlobMemory *BorrowBlobMemory(int use_count, BlobMemorySizeInfo &size_info, bool use_new_memory = false);
    void RefundBlobMemory(BlobMemory *blob_memory);
    int GetAllBlobMemorySize();
    Status AssignAllBlobMemory(MemoryAssignStrategy &strategy);
    virtual void ClearBlobMemoryPool();
    AbstractDevice *GetDevice();
protected:
    AbstractDevice *device_ = nullptr;
    void ReleaseBlobMemoryNodeList(BlobMemoryNode *list_header);

private:
    BlobMemoryPool(const BlobMemoryPool &);
    BlobMemoryPool &operator=(const BlobMemoryPool &);

    virtual BlobMemory *CreateBlobMemory(int use_count, BlobMemorySizeInfo &size_info)              = 0;
    virtual BlobMemoryNode *GetBlobMemoryNodeListHeader(DataType data_type)                         = 0;
    virtual void SetBlobMemoryNodeListHeader(DataType data_type, BlobMemoryNode *new_header)        = 0;
    virtual int64_t ResolveBlobMemoryNodeBytesDiff(BlobMemorySizeInfo &size_info, BlobMemoryNode *node) = 0;

    void CalculateAllBlobMemorySize();
    // extract the closest BlobMemoryNode from BlobMemoryNode list
    virtual BlobMemoryNode *ExtractNearestBlobMemoryNode(BlobMemorySizeInfo &size_info);

    int all_blob_memory_size_ = 0;;
    std::set<BlobMemory *> blob_memory_library_ = {};
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_MEMORY_MANAGER_BLOB_MEMORY_POOL_H_
