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

#ifndef TNN_SOURCE_TNN_MEMORY_MANAGER_BLOB_2D_MEMORY_H_
#define TNN_SOURCE_TNN_MEMORY_MANAGER_BLOB_2D_MEMORY_H_

#include <map>

#include "tnn/memory_manager/blob_memory.h"

namespace TNN_NS {

class Blob2DMemory : public BlobMemory {
public:
    explicit Blob2DMemory(AbstractDevice *device, BlobMemorySizeInfo &size_info, int use_count = 0);
    virtual ~Blob2DMemory();

    virtual void UpdateBlobMemorySizeInfo(BlobMemorySizeInfo info);

private:
    Blob2DMemory(const Blob2DMemory &);
    Blob2DMemory &operator=(const Blob2DMemory &);
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_MEMORY_MANAGER_BLOB_2D_MEMORY_H_
