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

#include "tnn/core/blob_manager.h"

#include <algorithm>
#include <cstring>
#include <set>

#include "tnn/memory_manager/blob_memory_pool_factory.h"
#include "tnn/memory_manager/blob_memory_size_info.h"
#include "tnn/memory_manager/memory_mode_state_factory.h"
#include "tnn/memory_manager/memory_seperate_assign_strategy.h"
#include "tnn/memory_manager/memory_unify_assign_strategy.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

BlobManager::BlobManager(AbstractDevice *device) {
    device_            = device;
    blob_memory_pool_  = BlobMemoryPoolFactory::CreateBlobMemoryPool(device);
    net_structure_     = nullptr;
    memory_mode_state_ = nullptr;
}

BlobManager::~BlobManager() {
    DeInit();

    if (blob_memory_pool_ != NULL) {
        delete blob_memory_pool_;
        blob_memory_pool_ = NULL;
    }
}

Status BlobManager::Init(NetworkConfig &config, NetStructure *net_structure, InputShapesMap inputs_shape_map,
                         DataType input_data_type) {
    if (net_structure->blobs.empty()) {
        LOGE("net_structure blobs is empty\n");
        return Status(TNNERR_PARAM_ERR, "net_structure blobs is empty");
    }

    net_structure_ = net_structure;
    
    auto instance_input_shapes_map = net_structure_->inputs_shape_map;
    if (instance_input_shapes_map.size() == 1 && inputs_shape_map.size() == 1) {
        // modify input shape if only one input, ignore the key
        instance_input_shapes_map.begin()->second = inputs_shape_map.begin()->second;
    } else {
        // modify input shape, only set invalid net input shape
        for (auto iter : inputs_shape_map) {
            if (instance_input_shapes_map.count(iter.first) > 0) {
                instance_input_shapes_map[iter.first] = iter.second;
            }
        }
    }

    config_            = config;
    init_thread_id_    = std::this_thread::get_id();
    memory_mode_state_ = MemoryModeStateFactory::CreateMemoryModeState(config.share_memory_mode);

    // get the maximum dimension of all inputs
    int input_dims = 0;
    for (auto blob_dims : instance_input_shapes_map) {
        int dims   = (int)blob_dims.second.size();
        input_dims = std::max(input_dims, dims);
    }

    // only supports dims >=4 .
    if (input_dims < 4) {
        LOGE("invalid input shape\n");
        return Status(TNNERR_PARAM_ERR, "invalid input shape");
    }

    for (auto node_name : net_structure_->blobs) {
        BlobDesc desc;
        desc.device_type = config.device_type;
        desc.data_type   = DATA_TYPE_FLOAT;
        desc.name        = node_name;
        // set to the specified data format
        if (config.data_format != DATA_FORMAT_AUTO) {
            desc.data_format = config.data_format;
        }

        // check whether the input_shape is defined or not.
        if (instance_input_shapes_map.count(node_name) > 0) {
            desc.dims = instance_input_shapes_map[node_name];
        }
        BlobHandle handle;
        blobs_[node_name] = new Blob(desc, handle);
    }

    // intput blobs
    for (auto iter : instance_input_shapes_map) {
        std::string current_blob_name         = iter.first;
        Blob *current_blob                    = blobs_[current_blob_name];
        current_blob->GetBlobDesc().data_type = input_data_type;
        input_blobs_[current_blob_name]       = current_blob;
    }

    // output blobs
    std::set<std::string> &output_blob_names = net_structure_->outputs;
    for (auto name : output_blob_names) {
        Blob *blob          = blobs_[name];
        output_blobs_[name] = blob;
    }

    return TNN_OK;
}

/*
 *  This function allocates the memory for all blobs.
 *  The memory size is calclucated by each Device according to data_type \
 *  and data format.
 *  The size may be different for different devices.
 */
Status BlobManager::AllocateBlobMemory() {
    const auto &input_shapes_map = net_structure_->inputs_shape_map;

    for (auto iter : input_shapes_map) {
        std::string current_blob_name = iter.first;
        Blob *current_blob            = blobs_[current_blob_name];
        // todo. need refactor
        BlobMemorySizeInfo info = device_->Calculate(current_blob->GetBlobDesc());
        if (info.dims.size() > 1 && config_.share_memory_mode != SHARE_MEMORY_MODE_DEFAULT) {
            return Status(TNNERR_SHARE_MEMORY_MODE_NOT_SUPPORT, "share_memory_mode option is unsupported");
        }
        int use_count           = 1;
        BlobMemory *blob_memory = NULL;
        blob_memory             = blob_memory_pool_->BorrowBlobMemory(use_count, info, true);
        blob_memory_mapping_.insert(std::make_pair(current_blob, blob_memory));
    }

    /*
     *  We reuse blob memory of the previos layers if it is not referenced.
     *  So, a use_count is calculated here.
     */
    for (int layer_index = 0; layer_index < net_structure_->layers.size(); layer_index++) {
        LayerInfo *layer_info = net_structure_->layers[layer_index].get();
        // allocating blob memory for every out nodes of this layer
        for (auto current_blob_name : layer_info->outputs) {
            Blob *current_blob = blobs_[current_blob_name];
            // ASSERT(current_blob->count() > 0);
            if (DimsVectorUtils::Count(current_blob->GetBlobDesc().dims) <= 0) {
                LOGE("Got empty blob, name:%s\n", current_blob_name.c_str());
                return Status(TNNERR_LAYER_ERR, "blob dims is invaid");
            }

            if (blob_memory_mapping_.find(current_blob) == blob_memory_mapping_.end()) {
                // calculate the use count of this blob
                int use_count = GetBlobUseCount(layer_index, current_blob_name);

                BlobMemorySizeInfo info = device_->Calculate(current_blob->GetBlobDesc());
                // find an available BlobMemory
                BlobMemory *blob_memory = blob_memory_pool_->BorrowBlobMemory(use_count, info, false);
                blob_memory_mapping_.insert(std::make_pair(current_blob, blob_memory));
            }
        }

        // refund the input blob memory
        for (auto current_blob_name : layer_info->inputs) {
            Blob *current_blob = blobs_[current_blob_name];
            if (input_shapes_map.count(current_blob_name) == 0) {
                std::map<Blob *, BlobMemory *>::const_iterator blob_memory_iter =
                    blob_memory_mapping_.find(current_blob);
                ASSERT(blob_memory_iter->second->GetUseCount() > 0);
                blob_memory_iter->second->DecrementUseCount();
                if (blob_memory_iter->second->GetUseCount() == 0) {
                    blob_memory_pool_->RefundBlobMemory(blob_memory_iter->second);
                }
            }
        }
    }
    
    Status status = TNN_OK;
    
    do {
        if (config_.share_memory_mode == SHARE_MEMORY_MODE_DEFAULT) {
            // The default strategy allocated the blob memory seperately.
            MemorySeperateAssignStrategy strategy;
            status = blob_memory_pool_->AssignAllBlobMemory(strategy);
            BREAK_IF(status != TNN_OK);
            BindBlobMemory();
        } else if (config_.share_memory_mode == SHARE_MEMORY_MODE_SHARE_ONE_THREAD) {
            // The share_on_thread strategy may share memory of different models-
            // whithin the same thread.
            int forward_memory_size   = blob_memory_pool_->GetAllBlobMemorySize();
            SharedMemory share_memory = SharedMemoryManager::GetSharedMemory(forward_memory_size, init_thread_id_, device_,
                                                                            config_.device_id, this, status);
            BREAK_IF(status != TNN_OK);
            MemoryUnifyAssignStrategy strategy(share_memory.shared_memory_data);
            status = blob_memory_pool_->AssignAllBlobMemory(strategy);
            BREAK_IF(status != TNN_OK);
            BindBlobMemory();
        }
    } while (0);

    return status;
}

/*
 * This function calculate the use count of the given blob.
 * output layer is regarded as an additional reference.
 */
int BlobManager::GetBlobUseCount(int layer_index, std::string current_blob_name) {
    int use_count                            = 0;
    std::set<std::string> &output_blob_names = net_structure_->outputs;
    for (int next_layer_id = layer_index + 1; next_layer_id != net_structure_->layers.size(); ++next_layer_id) {
        LayerInfo *next_layer_info = net_structure_->layers[next_layer_id].get();
        for (auto blob_name : next_layer_info->inputs) {
            if (strcmp(current_blob_name.c_str(), blob_name.c_str()) == 0) {
                ++use_count;
            }
        }
    }

    bool is_output_layer = output_blob_names.count(current_blob_name) > 0;
    if (use_count == 0 || is_output_layer) {
        use_count += 1;
    }
    return use_count;
}

Status BlobManager::DeInit() {
    if (config_.share_memory_mode == SHARE_MEMORY_MODE_SHARE_ONE_THREAD) {
        SharedMemoryManager::ReleaseSharedMemory(init_thread_id_, device_, config_.device_id, this);
    }

    for (auto blob : blobs_) {
        delete blob.second;
    }

    if (memory_mode_state_ != NULL) {
        delete memory_mode_state_;
        memory_mode_state_ = NULL;
    }
    return TNN_OK;
}

void BlobManager::OnSharedForwardMemoryChanged(void *memory) {
    MemoryUnifyAssignStrategy strategy(memory);
    blob_memory_pool_->AssignAllBlobMemory(strategy);
    BindBlobMemory();
}

/*
 * Blob memory may be allocated by the user.
 * The total size required is given by GetAllBlobMemorySize().
 */
Status BlobManager::SetForwardMemory(void *memory) {
    if (config_.share_memory_mode != SHARE_MEMORY_MODE_SET_FROM_EXTERNAL) {
        return Status(TNNERR_NOT_SUPPORT_SET_FORWARD_MEM, "set memory from external is unsupported");
    }
    MemoryUnifyAssignStrategy strategy(memory);
    auto status = blob_memory_pool_->AssignAllBlobMemory(strategy);
    if (status == TNN_OK) {
        BindBlobMemory();
    }
    return status;
}

void BlobManager::BindBlobMemory() {
    memory_mode_state_->SetMemoryAllocatedFlag();
    // bind every blob_memory's data_ into every blob's data
    for (auto iter : blob_memory_mapping_) {
        iter.first->SetHandle(iter.second->GetHandle());
    }
}

int BlobManager::GetAllBlobMemorySize() {
    return blob_memory_pool_->GetAllBlobMemorySize();
}

Status BlobManager::GetAllInputBlobs(BlobMap &blobs) {
    blobs = input_blobs_;
    return TNN_OK;
}

Status BlobManager::GetAllOutputBlobs(BlobMap &blobs) {
    blobs = output_blobs_;
    return TNN_OK;
}

Blob *BlobManager::GetBlob(std::string name) {
    return blobs_[name];
}

void BlobManager::ReplaceBlob(std::string name, Blob *new_blob) {
    if (blobs_.find(name) != blobs_.end()) {
        auto ori_blob = blobs_[name];
        if (ori_blob) {
            delete ori_blob;
        }
    }
    blobs_[name] = new_blob;

    if (input_blobs_.find(name) != input_blobs_.end())
        input_blobs_[name] = new_blob;

    if (output_blobs_.find(name) != output_blobs_.end())
        output_blobs_[name] = new_blob;
}

Status BlobManager::CheckBlobMemoryState() {
    return memory_mode_state_->GetStatus();
}

}  // namespace TNN_NS
