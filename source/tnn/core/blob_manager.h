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

#ifndef TNN_SOURCE_TNN_CORE_BLOB_MANAGER_H_
#define TNN_SOURCE_TNN_CORE_BLOB_MANAGER_H_

#include <map>
#include <memory>
#include <string>
#include <thread>

#include "tnn/core/abstract_device.h"
#include "tnn/core/blob.h"
#include "tnn/core/common.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/net_structure.h"
#include "tnn/memory_manager/blob_memory.h"
#include "tnn/memory_manager/blob_memory_pool.h"
#include "tnn/memory_manager/memory_assign_strategy.h"
#include "tnn/memory_manager/memory_mode_state.h"
#include "tnn/memory_manager/shared_memory_manager.h"

namespace TNN_NS {

class BlobManager : public ISharedMemoryChangeListener {
public:
    // @brief BlobManager constructor
    explicit BlobManager(AbstractDevice *device);

    // @brief BlobManager destructor
    ~BlobManager();

    // @brief InitBlobs init blobs
    // @param structure net structure
    virtual Status Init(NetworkConfig &config, NetStructure *net_structure, InputShapesMap inputs_shape_map,
                DataType input_data_type);

    // @brief DeInit release Init create resource
    Status DeInit();

    // @brief GetBlobs get blob by name
    // @param name blob name
    Blob *GetBlob(std::string name);

    // @brief check blob memory state for different share memory mode
    Status CheckBlobMemoryState();

    // @brief set blob forward memory
    virtual Status SetForwardMemory(void *memory);

    // @brief get all input blobs
    // @param blobs blob map
    virtual Status GetAllInputBlobs(BlobMap &blobs);

    // @brief get all output blobs
    // @param blobs blob map
    virtual Status GetAllOutputBlobs(BlobMap &blobs);

    // @brief AllocateBlobMemory for blob with flag
    virtual Status AllocateBlobMemory(int flag = DATA_FLAG_CHANGE_ALWAYS);

    // @brief OnSharedForwardMemoryChanged for share memory change observer
    virtual void OnSharedForwardMemoryChanged(void *memory);

    // @brief get all blob memory size
    int GetAllBlobMemorySize();

    // @brief replace blob with new_blob, and delete the original blob if exist
    void ReplaceBlob(std::string name, Blob *new_blob);

protected:
    void BindBlobMemory();
    int GetBlobUseCount(int layer_index, std::string current_blob_name);

    NetworkConfig config_;
    NetStructure *net_structure_;
    // dimension-memory pool
    std::map<int, BlobMemoryPool *> blob_memory_pool_map_;
    AbstractDevice *device_;
    BlobMap input_blobs_;
    BlobMap output_blobs_;
    std::shared_ptr<MemoryAssignStrategy> strategy_;
    std::map<std::string, Blob *> blobs_;
    std::map<Blob *, BlobMemory *> blob_memory_mapping_;
    bool shared_memory_allocated_;

    std::thread::id init_thread_id_;
    MemoryModeState *memory_mode_state_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_CORE_BLOB_MANAGER_H_
