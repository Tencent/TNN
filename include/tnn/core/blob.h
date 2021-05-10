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

#ifndef TNN_INCLUDE_TNN_CORE_BLOB_H_
#define TNN_INCLUDE_TNN_CORE_BLOB_H_

#include <cstdint>
#include <map>
#include <string>

#include "tnn/core/common.h"
#include "tnn/core/macro.h"
#include "tnn/utils/dims_vector_utils.h"

#pragma warning(push)
#pragma warning(disable : 4251)

namespace TNN_NS {

//@brief BlobDesc blob data info
struct PUBLIC BlobDesc {
    // device_type describes device cpu, gpu, ...
    DeviceType device_type = DEVICE_NAIVE;
    // data_type describes data precision fp32, in8, ...
    DataType data_type = DATA_TYPE_FLOAT;
    // data_format describes data order nchw, nhwc, ...
    DataFormat data_format = DATA_FORMAT_AUTO;
    // DimsVector describes data dims
    DimsVector dims;
    // name describes the blob name
    std::string name = "";
    
    std::string description(bool all_message = false);
};

struct PUBLIC BlobHandle {
    void *base            = NULL;
    uint64_t bytes_offset = 0;
};

class BlobImpl;

// @brief Blob tnn data store and transfer interface.
class PUBLIC Blob {
public:
    //@brief create blob with blob descript
    explicit Blob(BlobDesc desc);

    Blob(BlobDesc desc, bool alloc_memory);

    //@brief create Blob with blob descript and data handle
    Blob(BlobDesc desc, BlobHandle handle);

    virtual ~Blob();

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

    int GetFlag();

    void SetFlag(int flag);
private: 
    BlobImpl* impl_;
};

// InputShapeMap input reshape info
using InputShapesMap   = std::map<std::string, DimsVector>;
using InputDataTypeMap = std::map<std::string, DataType>;

using BlobMap = std::map<std::string, Blob *>;

}  // namespace TNN_NS

#pragma warning(pop)

#endif  // TNN_INCLUDE_TNN_CORE_BLOB_H_
