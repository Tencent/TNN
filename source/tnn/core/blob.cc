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

namespace TNN_NS {

Blob::Blob(BlobDesc desc) {
    desc_ = desc;
}

Blob::Blob(BlobDesc desc, BlobHandle handle) {
    desc_   = desc;
    handle_ = handle;
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
    handle_ = handle;
}

}  // namespace TNN_NS
