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

#ifndef TNN_SOURCE_TNN_EXTERN_WRAPPER_FOREIGN_BLOB_H_
#define TNN_SOURCE_TNN_EXTERN_WRAPPER_FOREIGN_BLOB_H_

#include <cstdint>
#include <map>
#include <memory>
#include <vector>
#include <string>

#include "tnn/core/common.h"
#include "tnn/core/macro.h"
#include "tnn/core/blob.h"
#include "tnn/core/status.h"
#include "tnn/extern_wrapper/foreign_tensor.h"

namespace TNN_NS {


// @brief ForeignBlob holds foreign network tensor.
class ForeignBlob : public Blob {
public:
    //@brief create foreignblob with blob descript
    explicit ForeignBlob(BlobDesc desc);

    //@brief create foreignBlob with blob descript only
    ForeignBlob(BlobDesc desc, bool alloc_memory);

    //@brief create foreignBlob with blob descript and data handle
    ForeignBlob(BlobDesc desc, BlobHandle handle);

    //@brief create foreignBlob with blob only
    ForeignBlob(Blob * blob);

    ~ForeignBlob();    

    //@brief get the ForeignTensor
    std::shared_ptr<ForeignTensor> GetForeignTensor();

    //@brief set the ForeignTensor
    Status SetForeignTensor(std::shared_ptr<ForeignTensor> foreign_tensor);

protected:

    std::shared_ptr<ForeignTensor> foreign_tensor_;

};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_EXTERN_WRAPPER_FOREIGN_BLOB_H_
