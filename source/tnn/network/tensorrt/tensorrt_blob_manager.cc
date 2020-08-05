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

#include "tnn/network/tensorrt/tensorrt_blob_manager.h"

namespace TNN_NS {

TensorRTBlobManager::TensorRTBlobManager(AbstractDevice *device) : BlobManager(device) {
    engine_ = nullptr;
    context_blob_ = nullptr;
}

TensorRTBlobManager::~TensorRTBlobManager() {
}

Status TensorRTBlobManager::AllocateBlobMemory() {
    // input
    for (auto iter : input_blobs_) {
        Blob *current_blob = iter.second;
        BlobMemorySizeInfo info = device_->Calculate(current_blob->GetBlobDesc());
        if (info.dims.size() > 1 && config_.share_memory_mode != SHARE_MEMORY_MODE_DEFAULT) {
            return Status(TNNERR_SHARE_MEMORY_MODE_NOT_SUPPORT, "share_memory_mode option is unsupported");
        }
        int use_count = 1;
        BlobMemory *blob_memory = nullptr;
        blob_memory = blob_memory_pool_->BorrowBlobMemory(use_count, info, true);
        blob_memory_mapping_.insert(std::make_pair(current_blob, blob_memory));
    }

    // context
    {
        size_t context_memory_size = engine_->getDeviceMemorySize();
        BlobDesc desc;
        desc.device_type = config_.device_type;
        desc.data_type = DATA_TYPE_FLOAT;
        desc.name = "tensorrt_context_";
        // TODO(johnzlli) need refactor
        desc.dims = { int(context_memory_size / sizeof(float)) };
        BlobHandle handle;
        context_blob_ = new Blob(desc, handle);
        blobs_[desc.name] = context_blob_;
        BlobMemorySizeInfo info = device_->Calculate(context_blob_->GetBlobDesc());
        BlobMemory *blob_memory = nullptr;
        int use_count = 1;
        blob_memory = blob_memory_pool_->BorrowBlobMemory(use_count, info, true);
        blob_memory_mapping_.insert(std::make_pair(context_blob_, blob_memory));
    }

    // output
    for (auto iter : output_blobs_) {
        Blob *current_blob = iter.second;
        BlobMemorySizeInfo info = device_->Calculate(current_blob->GetBlobDesc());
        if (info.dims.size() > 1 && config_.share_memory_mode != SHARE_MEMORY_MODE_DEFAULT) {
            return Status(TNNERR_SHARE_MEMORY_MODE_NOT_SUPPORT, "share_memory_mode option is unsupported");
        }
        int use_count = 1;
        BlobMemory *blob_memory = nullptr;
        blob_memory = blob_memory_pool_->BorrowBlobMemory(use_count, info, true);
        blob_memory_mapping_.insert(std::make_pair(current_blob, blob_memory));
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

void* TensorRTBlobManager::GetContextMemory() {
    return context_blob_->GetHandle().base;
}

void TensorRTBlobManager::SetEngine(nvinfer1::ICudaEngine* engine) {
    this->engine_ = engine;
}

}  //  TNN_NS