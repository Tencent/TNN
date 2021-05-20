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

#ifndef TNN_INCLUDE_TNN_CORE_BLOB_IMPL_H_
#define TNN_INCLUDE_TNN_CORE_BLOB_IMPL_H_

#include <cstdint>
#include <map>
#include <string>

#include "tnn/core/blob.h"
#include "tnn/core/common.h"
#include "tnn/core/macro.h"

#pragma warning(push)
#pragma warning(disable : 4251)

namespace TNN_NS {

// @brief BlobImpl tnn data store and transfer interface.
class PUBLIC BlobImpl {
public:
    //@brief create blob with blob descript
    explicit BlobImpl(BlobDesc desc);

    BlobImpl(BlobDesc desc, bool alloc_memory);

    //@brief create Blob with blob descript and data handle
    BlobImpl(BlobDesc desc, BlobHandle handle);

    virtual ~BlobImpl();

    //@brief return blob desc
    BlobDesc &GetBlobDesc();

    //@brief set blob description
    //@param desc blob description
    void SetBlobDesc(BlobDesc desc);

    //@brief return handle to the stored data
    BlobHandle GetHandle();

    //@brief set blob handle
    //@param handle to the stored data
    void SetHandle(BlobHandle handle);

    //@brief allocate blob handle in forward
    bool NeedAllocateInForward();
    
    //@brief check if it is constant
    bool IsConstant();

    //@brief get blob flag
    int GetFlag();
  
    //@brief set blob flag
    void SetFlag(int flag);

private:
    BlobDesc desc_;
    BlobHandle handle_;
    bool alloc_memory_;
    //0: data alwalys change
    //1: data change if shape differ
    //2: data never change
    int flag_ = DATA_FLAG_CHANGE_ALWAYS;
};

}  // namespace TNN_NS

#pragma warning(pop)

#endif  // TNN_INCLUDE_TNN_CORE_BLOB_IMPL_H_
