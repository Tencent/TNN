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

#ifndef TNN_INCLUDE_TNN_CORE_BLOB_INT8_H_
#define TNN_INCLUDE_TNN_CORE_BLOB_INT8_H_

#include "tnn/core/blob.h"
#include "tnn/interpreter/layer_resource.h"

namespace TNN_NS {

// @brief Blob tnn data store and transfer interface.
class BlobInt8 : public Blob {
public:
    //@brief create BlobInt8 with blob descript
    explicit BlobInt8(BlobDesc desc);

    //@brief create BlobInt8 with blob descript and data handle
    BlobInt8(BlobDesc desc, BlobHandle handle);

    //@brief get layer int8 resources
    IntScaleResource *GetIntResource();

    //@brief set layer int8 resources
    void SetIntResource(IntScaleResource *resource);

private:
    IntScaleResource *resource_ = nullptr;
};

}  // namespace TNN_NS

#endif  // TNN_INCLUDE_TNN_CORE_BLOB_INT8_H_
