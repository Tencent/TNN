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

#include <fstream>
#include <string>

#include "tnn/device/cuda/cuda_device.h"
#include "tnn/network/tensorrt/tensorrt_blob_manager.h"
#include "tnn/network/tensorrt/tensorrt_tensor.h"
#include "tnn/memory_manager/blob_memory_pool_factory.h"
#include "tnn/memory_manager/blob_memory_size_info.h"
#include "tnn/memory_manager/memory_mode_state_factory.h"
#include "tnn/memory_manager/memory_seperate_assign_strategy.h"
#include "tnn/memory_manager/memory_unify_assign_strategy.h"

namespace TNN_NS {

TensorRTBlobManager::TensorRTBlobManager(AbstractDevice *device) : BlobManager(device) {
}

TensorRTBlobManager::~TensorRTBlobManager() {
}

Status TensorRTBlobManager::Init(NetworkConfig &config, NetStructure *net_structure, InputShapesMap inputs_shape_map,
        DataType input_data_type) {
    if (net_structure->blobs.empty()) {
        LOGE("net_structure blobs is empty\n");
        return Status(TNNERR_PARAM_ERR, "net_structure blobs is empty");
    }

    net_structure_ = net_structure;
    // modify input shape, only set invalid net input shape
    auto instance_input_shapes_map = net_structure_->inputs_shape_map;
    for (auto iter : inputs_shape_map) {
        if (instance_input_shapes_map.count(iter.first) > 0) {
            instance_input_shapes_map[iter.first] = iter.second;
        }
    }

    config_            = config;
    init_thread_id_    = std::this_thread::get_id();
    memory_mode_state_ = MemoryModeStateFactory::CreateMemoryModeState(SHARE_MEMORY_MODE_DEFAULT);

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
        blobs_[node_name] = new ForeignBlob(desc, handle);
        auto tensorrtTensor = std::make_shared<TensorRTTensor>();
        dynamic_cast<ForeignBlob*>(blobs_[node_name])->SetForeignTensor(tensorrtTensor);
    }

    // input blobs
    for (auto iter : instance_input_shapes_map) {
        std::string current_blob_name         = iter.first;
        Blob *current_blob                    = blobs_[current_blob_name];
        current_blob->GetBlobDesc().data_type = input_data_type;
        input_blobs_[current_blob_name]       = current_blob;
    }

    // input data types
    const auto& input_data_type_map = net_structure->input_data_type_map;
    for (auto iter : instance_input_shapes_map) {
        std::string current_blob_name         = iter.first;
        Blob *current_blob                    = blobs_[current_blob_name];
        if (input_data_type_map.find(current_blob_name) != input_data_type_map.end()) {
            current_blob->GetBlobDesc().data_type = input_data_type_map.find(current_blob_name)->second;
        }
        input_blobs_[current_blob_name]       = current_blob;
    }

    // output blobs
    std::set<std::string> &output_blob_names = net_structure_->outputs;
    for (auto name : output_blob_names) {
        Blob *blob = blobs_[name];
        if (std::find(net_structure_->blobs.begin(), net_structure_->blobs.end(), name) != net_structure_->blobs.end()) {
            output_blobs_[name] = blob;
        }
    }

    return TNN_OK;
}

Status TensorRTBlobManager::AllocateBlobMemory(int flag) {
    // input
    for (auto iter : input_blobs_) {
        Blob *current_blob = iter.second;
        BlobMemorySizeInfo info = device_->Calculate(current_blob->GetBlobDesc());
        info.data_type = DATA_TYPE_FLOAT;
        int use_count = 1;
        BlobMemory *blob_memory = nullptr;
        blob_memory = blob_memory_pool_map_[info.dims.size()]->BorrowBlobMemory(use_count, info, true);
        blob_memory_mapping_.insert(std::make_pair(current_blob, blob_memory));
    }

    // output
    for (auto iter : output_blobs_) {
        Blob *current_blob = iter.second;
        BlobMemorySizeInfo info = device_->Calculate(current_blob->GetBlobDesc());
        info.data_type = DATA_TYPE_FLOAT;
        int use_count = 1;
        BlobMemory *blob_memory = nullptr;
        blob_memory = blob_memory_pool_map_[info.dims.size()]->BorrowBlobMemory(use_count, info, true);
        blob_memory_mapping_.insert(std::make_pair(current_blob, blob_memory));
    }

    Status status = TNN_OK;

    do {
            // ignore share memory mode, allocated the blob memory separately.
            MemorySeperateAssignStrategy strategy;
            for (auto blob_memory_pool_iter : blob_memory_pool_map_) {
                status = blob_memory_pool_iter.second->AssignAllBlobMemory(strategy);
                BREAK_IF(status != TNN_OK);
            }
            BREAK_IF(status != TNN_OK);
            BindBlobMemory();
    } while (0);

    return status;
}

Status TensorRTBlobManager::MemAlloc(void **ptr, size_t size) {
    Status ret = dynamic_cast<CudaDevice*>(device_)->Allocate(ptr, size);
    return ret;
}

Status TensorRTBlobManager::MemFree(void* ptr) {
    Status ret = dynamic_cast<CudaDevice*>(device_)->Free(ptr);
    return ret;
}

}  //  TNN_NS
