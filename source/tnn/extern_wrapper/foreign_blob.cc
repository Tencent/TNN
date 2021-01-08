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

#include "tnn/extern_wrapper/foreign_blob.h"

#include <memory>

#include "tnn/core/blob.h"
#include "tnn/extern_wrapper/foreign_tensor.h"

namespace TNN_NS {

//@brief create foreignBlob with blob only
ForeignBlob::ForeignBlob(Blob* blob): Blob(blob->GetBlobDesc(), blob->GetHandle()) {
    foreign_tensor_ = std::make_shared<ForeignTensor>();
}

//@brief create foreignBlob with blob descript only
ForeignBlob::ForeignBlob(BlobDesc desc, bool alloc_memory): Blob(desc, alloc_memory) {
}

//@brief create foreignBlob with blob descript and data handle
ForeignBlob::ForeignBlob(BlobDesc desc, BlobHandle handle): Blob(desc, handle) {
}

ForeignBlob::~ForeignBlob() {
}

//@brief get the ForeignTensor
std::shared_ptr<ForeignTensor> ForeignBlob::GetForeignTensor() {
    return foreign_tensor_;
}

//@brief set the ForeignTensor
Status ForeignBlob::SetForeignTensor(std::shared_ptr<ForeignTensor> foreign_tensor) {
    foreign_tensor_ = foreign_tensor;
    return TNN_OK;
}


}  // namespace TNN_NS
