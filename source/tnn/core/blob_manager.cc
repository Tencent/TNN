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
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/data_flag_utils.h"

namespace TNN_NS {

BlobManager::BlobManager(AbstractDevice *device) {
    device_            = device;
    // create 1d memory pool
    blob_memory_pool_map_[1] = BlobMemoryPoolFactory::CreateBlobMemoryPool(device);
    if (device->GetDeviceType() == DEVICE_OPENCL) {
        // create 2d memory pool
        blob_memory_pool_map_[2] = BlobMemoryPoolFactory::CreateBlobMemoryPool(device, 2);
    }
    net_structure_     = nullptr;
    memory_mode_state_ = nullptr;
}

BlobManager::~BlobManager() {
    DeInit();

    for (auto blob_memory_pool_iter : blob_memory_pool_map_) {
        if (blob_memory_pool_iter.second != nullptr) {
            delete blob_memory_pool_iter.second;
            blob_memory_pool_iter.second = nullptr;
        }
    }
}

static void UpdateDeviceInputDataFormat(NetworkConfig &config, Blob *input, const DeviceType &type,
                                        bool is_quantized_layer) {
    if (config.data_format != DATA_FORMAT_AUTO)
        return;
    if (type == DEVICE_ARM && is_quantized_layer) {
        input->GetBlobDesc().data_format = DATA_FORMAT_NHWC4;
    } else if (type == DEVICE_ARM || type == DEVICE_METAL) {
        input->GetBlobDesc().data_format = DATA_FORMAT_NC4HW4;
    } else if (type == DEVICE_OPENCL) {
        input->GetBlobDesc().data_format = DATA_FORMAT_NHC4W4;
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
    shared_memory_allocated_ = false;
    memory_mode_state_ = MemoryModeStateFactory::CreateMemoryModeState(config.share_memory_mode);

    // get the maximum dimension of all inputs
    int input_dims = 0;
    for (auto blob_dims : instance_input_shapes_map) {
        int dims   = (int)blob_dims.second.size();
        input_dims = std::max(input_dims, dims);
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
    bool is_quantized_net = GetQuantizedInfoFromNetStructure(net_structure);
    // input blobs
    const auto &input_data_type_map = net_structure->input_data_type_map;
    for (auto iter : instance_input_shapes_map) {
        auto current_blob_name = iter.first;
        if (blobs_.find(current_blob_name) == blobs_.end()) {
            continue;
        }
        auto current_blob = blobs_[current_blob_name];
        if (input_data_type_map.find(current_blob_name) != input_data_type_map.end()) {
            current_blob->GetBlobDesc().data_type = input_data_type_map.find(current_blob_name)->second;
        } else {
            current_blob->GetBlobDesc().data_type = input_data_type;
        }
        bool is_quantized_layer = false;
        if (is_quantized_net) {
            is_quantized_layer = IsQuantizedLayerFromInputName(net_structure, current_blob->GetBlobDesc().name);
        }
        UpdateDeviceInputDataFormat(config, current_blob, device_->GetDeviceType(), is_quantized_layer);
        input_blobs_[current_blob_name] = current_blob;
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
 *  The memory size is calculated by each Device according to data_type \
 *  and data format.
 *  The size may be different for different devices.
 */
Status BlobManager::AllocateBlobMemory(int flag) {
    const auto &input_shapes_map = net_structure_->inputs_shape_map;

    for (auto iter : input_shapes_map) {
        std::string current_blob_name = iter.first;
        Blob *current_blob            = blobs_[current_blob_name];
        if (current_blob->NeedAllocateInForward() ||
            DataFlagUtils::ChangeStatus(current_blob->GetFlag()) != DataFlagUtils::ChangeStatus(flag)) {
            continue;
        }
        // todo. need refactor
        BlobMemorySizeInfo info = device_->Calculate(current_blob->GetBlobDesc());
        if (info.dims.size() > 1 && config_.share_memory_mode != SHARE_MEMORY_MODE_DEFAULT) {
            return Status(TNNERR_SHARE_MEMORY_MODE_NOT_SUPPORT, "share_memory_mode option is unsupported");
        }
        int use_count           = 1;
        BlobMemory *blob_memory = NULL;
        blob_memory             = blob_memory_pool_map_[info.dims.size()]->BorrowBlobMemory(use_count, info, true);
        blob_memory_mapping_.insert(std::make_pair(current_blob, blob_memory));
    }

    /*
     *  We reuse blob memory of the previous layers if it is not referenced.
     *  So, a use_count is calculated here.
     */
    for (size_t layer_index = 0; layer_index < net_structure_->layers.size(); layer_index++) {
        LayerInfo *layer_info = net_structure_->layers[layer_index].get();
        // allocating blob memory for every out nodes of this layer
        for (auto current_blob_name : layer_info->outputs) {
            Blob *current_blob = blobs_[current_blob_name];
            if (current_blob->NeedAllocateInForward() ||
                DataFlagUtils::ChangeStatus(current_blob->GetFlag()) != DataFlagUtils::ChangeStatus(flag)) {
                continue;
            }
            
            // ASSERT(current_blob->count() > 0);
            if (DimsVectorUtils::Count(current_blob->GetBlobDesc().dims) < 0) {
                LOGE("Got empty blob, name:%s\n", current_blob_name.c_str());
                return Status(TNNERR_LAYER_ERR, "blob dims is invaid");
            }

            if (blob_memory_mapping_.find(current_blob) == blob_memory_mapping_.end()) {
                // calculate the use count of this blob
                int use_count = GetBlobUseCount(layer_index, current_blob_name);

                BlobMemorySizeInfo info = device_->Calculate(current_blob->GetBlobDesc());
                // find an available BlobMemory
                BlobMemory *blob_memory = blob_memory_pool_map_[info.dims.size()]->BorrowBlobMemory(use_count, info, false);
                blob_memory_mapping_.insert(std::make_pair(current_blob, blob_memory));
            }
        }

        // refund the input blob memory
        for (auto current_blob_name : layer_info->inputs) {
            Blob *current_blob = blobs_[current_blob_name];
            if (current_blob->NeedAllocateInForward() ||
                DataFlagUtils::ChangeStatus(current_blob->GetFlag()) != DataFlagUtils::ChangeStatus(flag)) {
                continue;
            }
            
            if (input_shapes_map.count(current_blob_name) == 0) {
                std::map<Blob *, BlobMemory *>::const_iterator blob_memory_iter =
                    blob_memory_mapping_.find(current_blob);
                ASSERT(blob_memory_iter->second->GetUseCount() > 0);
                blob_memory_iter->second->DecrementUseCount();
                if (blob_memory_iter->second->GetUseCount() == 0) {
                    int dimensions = blob_memory_iter->second->GetBlobMemorySizeInfo().dims.size();
                    blob_memory_pool_map_[dimensions]->RefundBlobMemory(blob_memory_iter->second);
                }
            }
        }
    }

    Status status = TNN_OK;

    do {
        if (config_.share_memory_mode == SHARE_MEMORY_MODE_DEFAULT) {
            // The default strategy allocated the blob memory separately.
            MemorySeperateAssignStrategy strategy;
            for (auto blob_memory_pool_iter : blob_memory_pool_map_) {
                status = blob_memory_pool_iter.second->AssignAllBlobMemory(strategy);
                BREAK_IF(status != TNN_OK);
            }
            BREAK_IF(status != TNN_OK);
            BindBlobMemory();
        } else if (config_.share_memory_mode == SHARE_MEMORY_MODE_SHARE_ONE_THREAD) {
            // The share_on_thread strategy may share memory of different models-
            // within the same thread.
            for (auto blob_memory_pool_iter : blob_memory_pool_map_) {
                int forward_memory_size   = blob_memory_pool_iter.second->GetAllBlobMemorySize();
                SharedMemory share_memory = SharedMemoryManager::GetSharedMemory(
                        forward_memory_size, init_thread_id_, device_,
                        config_.device_id, this, status);
                BREAK_IF(status != TNN_OK);
		shared_memory_allocated_ = true;
                MemoryUnifyAssignStrategy strategy(share_memory.shared_memory_data);
                status = blob_memory_pool_iter.second->AssignAllBlobMemory(strategy);
                BREAK_IF(status != TNN_OK);
            }
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
    for (size_t next_layer_id = layer_index + 1; next_layer_id != net_structure_->layers.size(); ++next_layer_id) {
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
    if(shared_memory_allocated_) {
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
    for (auto blob_memory_pool_iter : blob_memory_pool_map_) {
        blob_memory_pool_iter.second->AssignAllBlobMemory(strategy);
    }
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
    Status status = TNN_OK;
    for (auto blob_memory_pool_iter : blob_memory_pool_map_) {
        status = blob_memory_pool_iter.second->AssignAllBlobMemory(strategy);
    }
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
        // set blob data format to nchw when blob memory is 1d on opencl
        if (device_->GetDeviceType() == DEVICE_OPENCL &&
            iter.second->GetBlobMemorySizeInfo().dims.size() == 1) {
            auto desc = iter.first->GetBlobDesc();
            desc.data_format = DATA_FORMAT_NCHW;
            iter.first->SetBlobDesc(desc);
        }
    }
}

int BlobManager::GetAllBlobMemorySize() {
    int mem_size_all_blob = 0;
    for (auto blob_memory_pool_iter : blob_memory_pool_map_) {
        mem_size_all_blob += blob_memory_pool_iter.second->GetAllBlobMemorySize();
    }
    return mem_size_all_blob;
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
    auto iter = blobs_.find(name);
    if (iter != blobs_.end()) {
        return iter->second;
    }
    return nullptr;
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
