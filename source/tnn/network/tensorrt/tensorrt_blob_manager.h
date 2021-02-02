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

#ifndef TNN_SOURCE_TNN_NETWORK_TENSORRT_TENSORRT_BLOB_MANAGER_H_
#define TNN_SOURCE_TNN_NETWORK_TENSORRT_TENSORRT_BLOB_MANAGER_H_

#include "NvInfer.h"

#include "tnn/core/blob_manager.h"
#include "tnn/extern_wrapper/foreign_blob.h"

namespace TNN_NS {

class TensorRTBlobManager : public BlobManager {
public:
    // @brief TensorRTBlobManager constructor
    explicit TensorRTBlobManager(AbstractDevice *device);

    // @brief TensorRTBlobManager destructor
    ~TensorRTBlobManager();

    // @brief init tensorrt blobs
    // @param structure net structure
    virtual Status Init(NetworkConfig &config, NetStructure *net_structure, InputShapesMap inputs_shape_map,
                DataType input_data_type);

    // @brief AllocateBlobMemory
    Status AllocateBlobMemory(int flag = DATA_FLAG_CHANGE_ALWAYS) override;

    // @brief Allocate a memory buffer
    Status MemAlloc(void** ptr, size_t size);

    // @brief Free a memory buffer
    Status MemFree(void* ptr);
};

}  //  namespace TNN_NS

#endif  //  TNN_SOURCE_TNN_NETWORK_TENSORRT_TENSORRT_BLOB_MANAGER_H_
