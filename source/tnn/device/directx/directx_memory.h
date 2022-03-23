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

#ifndef TNN_SOURCE_TNN_DEVICE_DIRECTX_DIRECTX_MEMORY_H_
#define TNN_SOURCE_TNN_DEVICE_DIRECTX_DIRECTX_MEMORY_H_

#include <memory>

#define NOMINMAX
#include <d3dcommon.h>
#include <d3d11.h>
#undef LoadLibrary

#include "tnn/core/common.h"
#include "tnn/core/blob.h"

namespace TNN_NS {

namespace directx {

enum DirectXMemoryType { TNN_DX_BUFFER = 0, TNN_DX_TEXTURE = 1 };

// @brief DirectXMemory data store in platform and can be shared
class DirectXMemory {
public:
    // @brief create DirectXMemory with type
    // @param type: the type of memory
    explicit DirectXMemory(DirectXMemoryType type);

    ~DirectXMemory();

    // @brief get data pointer
    void* GetData() const;

    // @brief set data pointer
    void SetData(void* data_ptr, bool own_data = false);

    // @brief get memory type
    DirectXMemoryType GetMemoryType() const;

    // @brief set memory type
    void SetMemoryType(DirectXMemoryType type);

    // @brief Set data info from BlobDesc 
    void SetMemoryInfo(BlobDesc desc) {
        data_type_ = desc.data_type;
        data_format_ = desc.data_format;
        dims_ = desc.dims;
    }

    BlobDesc CreateBlobDesc() {
        BlobDesc desc;
        desc.data_type = data_type_;
        desc.data_format = data_format_;
        desc.dims = dims_;
        return desc;
    }

    // @brief Set data info 
    void SetMemoryInfo(DataType data_type, DataFormat data_format, DimsVector dims) {
        data_type_ = data_type;
        data_format_ = data_format;
        dims_ = dims;
    }

    // @brief Create SRV
    std::shared_ptr<ID3D11ShaderResourceView> GetSRV();

    // @brief Create UAV
    std::shared_ptr<ID3D11UnorderedAccessView> GetUAV();

    static std::shared_ptr<DirectXMemory> CreateRefMemoryFromBlob(Blob *);

private:
    // remove all assignment operator
    DirectXMemory(const DirectXMemory& memory)  = delete;
    DirectXMemory(const DirectXMemory&& memory) = delete;
    DirectXMemory& operator=(const DirectXMemory&) = delete;
    DirectXMemory& operator=(const DirectXMemory&&) = delete;

private:
    // data pointer
    void* data_ = nullptr;
    // memory type
    DirectXMemoryType mem_type_ = TNN_DX_TEXTURE;
    // own_data_ decide whether need to release data
    bool own_data_ = false;

    //memory data type
    DataType data_type_ = DATA_TYPE_FLOAT;
    // data_format describes data order nchw, nhwc, ...
    DataFormat data_format_ = DATA_FORMAT_AUTO;
    // DimsVector describes data dims
    DimsVector dims_;

    std::shared_ptr<ID3D11ShaderResourceView> srv_;
    std::shared_ptr<ID3D11UnorderedAccessView> uav_;
};

} // namespace directx

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_DIRECTX_DIRECTX_MEMORY_H_
