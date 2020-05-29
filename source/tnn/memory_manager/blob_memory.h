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

#ifndef TNN_SOURCE_TNN_MEMORY_MANAGER_BLOB_MEMORY_H_
#define TNN_SOURCE_TNN_MEMORY_MANAGER_BLOB_MEMORY_H_

#include "tnn/core/abstract_device.h"
#include "tnn/core/blob.h"

namespace TNN_NS {

class BlobMemory {
public:
    BlobMemory(AbstractDevice *device, BlobMemorySizeInfo &size_info, int use_count = 0);
    virtual ~BlobMemory();

    virtual void UpdateBlobMemorySizeInfo(BlobMemorySizeInfo info) = 0;
    BlobMemorySizeInfo GetBlobMemorySizeInfo() const;

    void SetUseCount(int use_count);
    int GetUseCount() const;
    bool DecrementUseCount();

    Status AllocateHandle();
    void SetHandleFromExternal(BlobHandle handle);
    BlobHandle GetHandle();

protected:
    BlobMemorySizeInfo size_info_;

private:
    BlobMemory(const BlobMemory &);
    BlobMemory &operator=(const BlobMemory &);

    AbstractDevice *device_;
    BlobHandle handle_;
    bool need_release_memory_;
    int use_count_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_MEMORY_MANAGER_BLOB_MEMORY_H_
