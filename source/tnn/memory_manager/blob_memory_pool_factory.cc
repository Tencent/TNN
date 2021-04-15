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

#include "tnn/memory_manager/blob_memory_pool_factory.h"
#include "tnn/memory_manager/blob_1d_memory_pool.h"
#include "tnn/memory_manager/blob_2d_memory_pool.h"

namespace TNN_NS {

BlobMemoryPool* BlobMemoryPoolFactory::CreateBlobMemoryPool(AbstractDevice* device, int dimensions) {
    if (DEVICE_OPENCL == device->GetDeviceType()) {
        if (dimensions == 2) {
            return new Blob2DMemoryPool(device);
        } else if (dimensions == 1) {
            return new Blob1DMemoryPool(device);
        } else {
            return nullptr;
        }
    } else {
        return new Blob1DMemoryPool(device);
    }
}

}  // namespace TNN_NS
