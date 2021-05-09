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

#include <iomanip>
#include <sstream>

#include "tnn/core/blob.h"
#include "tnn/core/blob_impl.h"

namespace TNN_NS {

std::string BlobDesc::description(bool all_message) {
    std::ostringstream os;
    //name
    os << "name: " <<name;

    //data type
    os << " data type: " << data_type;

    //shape
    os << " shape: [ " ;
    for (auto iter : dims) {
        os << iter << " " ;
    }
    os << "]";

    return os.str();
}

Blob::Blob(BlobDesc desc) {
    impl_ = new BlobImpl(desc);
}

Blob::Blob(BlobDesc desc, bool alloc_memory) {
    impl_ = new BlobImpl(desc, alloc_memory);
}

Blob::~Blob() {
    delete impl_;
}

Blob::Blob(BlobDesc desc, BlobHandle handle) {
    impl_ = new BlobImpl(desc, handle);
}

// Set the descriptor of the blob.
void Blob::SetBlobDesc(BlobDesc desc) {
    impl_->SetBlobDesc(desc);
}

// Get a reference of the descriptor of the blob.
BlobDesc &Blob::GetBlobDesc() {
    return impl_->GetBlobDesc();
}

// Get a copy of the handle of the blob.
BlobHandle Blob::GetHandle() {
    return impl_->GetHandle();
}

// Set the handle of the blob.
void Blob::SetHandle(BlobHandle handle) {
    impl_->SetHandle(handle);
}

//@brief allocate blob handle in forward
bool Blob::NeedAllocateInForward() {
    return impl_->NeedAllocateInForward();
}

bool Blob::IsConstant() {
    return impl_->IsConstant();
}

int Blob::GetFlag() {
    return impl_->GetFlag();
}

void Blob::SetFlag(int flag) {
    impl_->SetFlag(flag);
}

}  // namespace TNN_NS
