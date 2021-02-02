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

#ifndef TNN_SOURCE_TNN_MEMORY_MANAGER_BLOB_2D_MEMORY_POOL_H_
#define TNN_SOURCE_TNN_MEMORY_MANAGER_BLOB_2D_MEMORY_POOL_H_

#include <map>

#include "tnn/memory_manager/blob_memory_pool.h"

namespace TNN_NS {

class Blob2DMemoryPool : public BlobMemoryPool {
public:
    explicit Blob2DMemoryPool(AbstractDevice* device);
    virtual ~Blob2DMemoryPool();
    virtual void ClearBlobMemoryPool() override;

private:
    virtual BlobMemory* CreateBlobMemory(int use_count, BlobMemorySizeInfo& size_info) override;
    virtual BlobMemoryNode* GetBlobMemoryNodeListHeader(DataType data_type) override;
    virtual void SetBlobMemoryNodeListHeader(DataType data_type, BlobMemoryNode* new_header) override;
    virtual int ResolveBlobMemoryNodeBytesDiff(BlobMemorySizeInfo& size_info, BlobMemoryNode* node) override;
    // extract the closest BlobMemoryNode from BlobMemoryNode list for 2D memory
    virtual BlobMemoryNode* ExtractNearestBlobMemoryNode(BlobMemorySizeInfo& size_info) override;
    std::map<DataType, BlobMemoryNode*> blob_memory_list_header_map_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_MEMORY_MANAGER_BLOB_2D_MEMORY_POOL_H_
