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

#include "tnn/core/blob.h"
#include "tnn/core/abstract_device.h"
#include "tnn/memory_manager/blob_memory_size_info.h"

namespace TNN_NS {

Blob::Blob(BlobDesc desc) {
    desc_ = desc;
    alloc_memory_ = false;
}

Blob::Blob(BlobDesc desc, bool alloc_memory) {
    desc_ = desc;
    alloc_memory_ = alloc_memory;
    if(alloc_memory) {
        auto device = GetDevice(desc.device_type);
        if(device != NULL) {
            BlobMemorySizeInfo size_info = device->Calculate(desc);
            device->Allocate(&handle_.base, size_info);
        }
    }
}

Blob::~Blob() {
    if(alloc_memory_ && handle_.base != NULL) {
        auto device = GetDevice(desc_.device_type);
        if(device != NULL) {
            device->Free(handle_.base);
        }
    }
}

Blob::Blob(BlobDesc desc, BlobHandle handle) {
    desc_   = desc;
    handle_ = handle;
    alloc_memory_ = false;
}

// Set the descriptor of the blob.
void Blob::SetBlobDesc(BlobDesc desc) {
    desc_ = desc;
}

// Get a reference of the descriptor of the blob.
BlobDesc &Blob::GetBlobDesc() {
    return desc_;
}

// Get a copy of the handle of the blob.
BlobHandle Blob::GetHandle() {
    return handle_;
}

// Set the handle of the blob.
void Blob::SetHandle(BlobHandle handle) {
    if(alloc_memory_) {
        auto device = GetDevice(desc_.device_type);
        if(device != NULL) {
            device->Free(handle_.base);
        }
    }
    handle_ = handle;
    alloc_memory_ = false;
}

}  // namespace TNN_NS
