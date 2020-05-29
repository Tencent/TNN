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

#include "tnn/core/blob_int8.h"

#include "tnn/core/common.h"

namespace TNN_NS {

BlobInt8::BlobInt8(BlobDesc desc) : Blob(desc) {
    this->GetBlobDesc().data_type = DATA_TYPE_INT8;
}

BlobInt8::BlobInt8(BlobDesc desc, BlobHandle handle) : Blob(desc, handle) {
    this->GetBlobDesc().data_type = DATA_TYPE_INT8;
}

/*
 * The resource of Int8Blob stores the quantization scale.
 * Per-tensor scale and per-channel scale are supported Currently.
 */
IntScaleResource *BlobInt8::GetIntResource() {
    return resource_;
}

void BlobInt8::SetIntResource(IntScaleResource *resource) {
    resource_ = resource;
}

}  // namespace TNN_NS
