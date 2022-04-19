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

#include "tnn/device/directx/directx_memory.h"

#define NOMINMAX
#include <d3dcommon.h>
#include <d3d11.h>
#undef LoadLibrary

#include "tnn/device/directx/directx_runtime.h"
#include "tnn/device/directx/directx_util.h"
#include "tnn/device/directx/directx_device.h"
#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {

namespace directx {

DirectXMemory::DirectXMemory(DirectXMemoryType type) {
    mem_type_ = type;
}

DirectXMemory::~DirectXMemory() {
    if (own_data_ && data_ != nullptr) {
        IUnknown * p = static_cast<IUnknown *>(data_);
        p->Release();
        data_ = nullptr;
    }
}

void *DirectXMemory::GetData() const {
    return data_;
}

void DirectXMemory::SetData(void *data_ptr, bool own_data) {
    data_     = data_ptr;
    own_data_ = own_data;
}

DirectXMemoryType DirectXMemory::GetMemoryType() const {
    return mem_type_;
}

void DirectXMemory::SetMemoryType(DirectXMemoryType type) {
    mem_type_ = type;
}

// @brief Create SRV
std::shared_ptr<ID3D11ShaderResourceView> DirectXMemory::GetSRV() {
    if (!srv_) {

        auto tnn_device = dynamic_cast<DirectXDevice*>(GetDevice(DEVICE_DIRECTX));
        if (!tnn_device) {
            LOGE("Got null directx device");
            return std::shared_ptr<ID3D11ShaderResourceView>(nullptr);
        }

        auto device = tnn_device->GetID3DDevice();
        ID3D11ShaderResourceView * srv;

        if (TNN_DX_BUFFER == GetMemoryType()) {
            ID3D11Buffer * buffer = (ID3D11Buffer *) data_;

            D3D11_BUFFER_DESC descBuf = {};
            buffer->GetDesc( &descBuf );

            D3D11_SHADER_RESOURCE_VIEW_DESC desc = {};
            desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
            desc.BufferEx.FirstElement = 0;

            if ( descBuf.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS ) {
                // Raw Buffer
                desc.Format = DXGI_FORMAT_R32_TYPELESS;
                desc.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
                // TODO, get the element size from TNN blob data type and data format
                desc.BufferEx.NumElements = descBuf.ByteWidth / 4;
            } else if ( descBuf.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_STRUCTURED ) {
                // Structured Buffer
                desc.Format = DXGI_FORMAT_UNKNOWN;
                desc.BufferEx.NumElements = descBuf.ByteWidth / descBuf.StructureByteStride;
            } else {
                LOGE("unknow dx buffer type");
                return std::shared_ptr<ID3D11ShaderResourceView>(nullptr);
            }

            HRESULT hr = device->CreateShaderResourceView(buffer, &desc, &srv);
            if (FAILED(hr)) {
                LOGE("Create SRV from buffer failed, code: 0x%X", hr);  
                return std::shared_ptr<ID3D11ShaderResourceView>(nullptr);
            }
        } else {
            ID3D11Texture2D * texture = (ID3D11Texture2D*) data_;
            D3D11_TEXTURE2D_DESC descTexture= {};
            texture->GetDesc(&descTexture);

            D3D11_SHADER_RESOURCE_VIEW_DESC desc= {};
            desc.Texture2D.MipLevels = descTexture.MipLevels;
            desc.Texture2D.MostDetailedMip = 0;
            desc.Format = descTexture.Format;
            desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;

            HRESULT hr = device->CreateShaderResourceView(texture, &desc, &srv);
            if (FAILED(hr)) {
                LOGE("Create SRV from texture failed, code: 0x%X", hr);  
                return std::shared_ptr<ID3D11ShaderResourceView>(nullptr);
            }
        }

        srv_ = std::shared_ptr<ID3D11ShaderResourceView>(srv, [](ID3D11ShaderResourceView * p){ p->Release();});
    }
    return srv_;
}

// @brief Create UAV
std::shared_ptr<ID3D11UnorderedAccessView> DirectXMemory::GetUAV() {
    if (!uav_) {

        auto tnn_device = dynamic_cast<DirectXDevice*>(GetDevice(DEVICE_DIRECTX));
        if (!tnn_device) {
            LOGE("Got null directx device");
            return std::shared_ptr<ID3D11UnorderedAccessView>(nullptr);
        }

        auto device = tnn_device->GetID3DDevice();
        ID3D11UnorderedAccessView * uav;

        if (TNN_DX_BUFFER == GetMemoryType()) {
            ID3D11Buffer * buffer = (ID3D11Buffer *) data_;

            D3D11_BUFFER_DESC descBuf = {};
            buffer->GetDesc( &descBuf );

            D3D11_UNORDERED_ACCESS_VIEW_DESC desc = {};
            desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
            desc.Buffer.FirstElement = 0;

            if ( descBuf.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS ) {
                // Raw Buffer
                desc.Format = DXGI_FORMAT_R32_TYPELESS;
                desc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
                // TODO, get the element size from TNN blob data type and data format
                desc.Buffer.NumElements = descBuf.ByteWidth / 4;
            } else if ( descBuf.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_STRUCTURED ) {
                // Structured Buffer
                desc.Format = DXGI_FORMAT_UNKNOWN;
                desc.Buffer.NumElements = descBuf.ByteWidth / descBuf.StructureByteStride;
            } else {
                LOGE("unknow dx buffer type");
                return std::shared_ptr<ID3D11UnorderedAccessView>(nullptr);
            }

            HRESULT hr = device->CreateUnorderedAccessView(buffer, &desc, &uav);
            if (FAILED(hr)) {
                LOGE("Create UAV from buffer failed, code: 0x%X", hr);  
                return std::shared_ptr<ID3D11UnorderedAccessView>(nullptr);
            }
        } else {
            ID3D11Texture2D * texture = (ID3D11Texture2D*) data_;
            D3D11_TEXTURE2D_DESC descTexture= {};
            texture->GetDesc(&descTexture);

            D3D11_UNORDERED_ACCESS_VIEW_DESC desc= {};
            desc.Texture2D.MipSlice = 0;
            desc.Format = descTexture.Format;
            desc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;

            HRESULT hr = device->CreateUnorderedAccessView(texture, &desc, &uav);
            if (FAILED(hr)) {
                LOGE("Create SRV from texture failed, code: 0x%X", hr);  
                return std::shared_ptr<ID3D11UnorderedAccessView>(nullptr);
            }
        }

        uav_= std::shared_ptr<ID3D11UnorderedAccessView>(uav, [](ID3D11UnorderedAccessView* p){ p->Release();});
    }
    return uav_;
}

std::shared_ptr<DirectXMemory> DirectXMemory::CreateRefMemoryFromBlob(Blob * blob) {
    DirectXMemoryType mem_type = TNN_NS::directx::GetMemoryType(blob->GetBlobDesc());
    DirectXMemory * ret = new DirectXMemory(mem_type);
    ret->SetMemoryInfo(blob->GetBlobDesc());
    ret->SetData(blob->GetHandle().base, false);
    return std::shared_ptr<DirectXMemory>(ret);
}

std::shared_ptr<DirectXMemory> DirectXMemory::CreateBufferMemoryFromHost(
                                void * ptr, DimsVector dims, DataType data_type, DataFormat data_format) {
    
//    if (nullptr == ptr){
//        LOGE("Got nullptr");
//        return std::shared_ptr<DirectXMemory>(nullptr);
//    }
    
    auto tnn_device = dynamic_cast<DirectXDevice*>(GetDevice(DEVICE_DIRECTX));
    if (!tnn_device) {
        LOGE("Got null directx device");
        return std::shared_ptr<DirectXMemory>(nullptr);
    }

    BlobMemorySizeInfo size_info;
    size_info.data_type = data_type;
    // 1D to directx buffer, 2D to Texture2D
    size_info.dims = {DimsVectorUtils::Count(dims)};

    ID3D11Buffer * buf;
    Status ret = tnn_device->Allocate((void**)&buf, size_info);
    RETURN_VALUE_ON_NEQ(ret, TNN_OK, std::shared_ptr<DirectXMemory>(nullptr));

    if (nullptr != ptr) {
        auto d3d_context = tnn_device->GetID3DContext();
        if (!d3d_context) {
            LOGE("Got null d3d context");
            return std::shared_ptr<DirectXMemory>(nullptr);
        }
        d3d_context->UpdateSubresource(buf, 0, nullptr, ptr, 0, 0);
    }

    DirectXMemory * dx_mem = new DirectXMemory(TNN_DX_BUFFER);
    if (nullptr != ptr) {
        dx_mem->SetData(buf, true);
    } else {
        dx_mem->SetData(buf, false);
    }
    dx_mem->SetMemoryInfo(data_type, data_format, dims);
    return std::shared_ptr<DirectXMemory>(dx_mem);
}

std::shared_ptr<DirectXMemory> DirectXMemory::CreateTextureMemoryFromHost(
    void * ptr, DimsVector dims, DataType data_type, DataFormat data_format) {

    if (nullptr != ptr){
        LOGE("Cant create Texture2D with data");
        return std::shared_ptr<DirectXMemory>(nullptr);
    }

    auto tnn_device = dynamic_cast<DirectXDevice*>(GetDevice(DEVICE_DIRECTX));
    if (!tnn_device) {
        LOGE("Got null directx device");
        return std::shared_ptr<DirectXMemory>(nullptr);
    }

    BlobDesc desc;
    desc.dims = dims;
    desc.data_type = data_type;
    desc.data_format = data_format;

    BlobMemorySizeInfo size_info = Calculate2DCLImageMemorySize(desc);

    ID3D11Texture2D * buf;
    Status ret = tnn_device->Allocate((void**)&buf, size_info);
    RETURN_VALUE_ON_NEQ(ret, TNN_OK, std::shared_ptr<DirectXMemory>(nullptr));

    DirectXMemory * dx_mem = new DirectXMemory(TNN_DX_TEXTURE);
    dx_mem->SetData(buf, false);
    dx_mem->SetMemoryInfo(data_type, data_format, dims);
    return std::shared_ptr<DirectXMemory>(dx_mem);
}

std::shared_ptr<DirectXMemory> DirectXMemory::CreateTextureMemoryFromHost(
    void * ptr, DimsVector dims, int image_width, int image_height, DataType data_type, DataFormat data_format) {

    if (nullptr != ptr){
        LOGE("Cant create Texture2D with data");
        return std::shared_ptr<DirectXMemory>(nullptr);
    }

    auto tnn_device = dynamic_cast<DirectXDevice*>(GetDevice(DEVICE_DIRECTX));
    if (!tnn_device) {
        LOGE("Got null directx device");
        return std::shared_ptr<DirectXMemory>(nullptr);
    }

    BlobMemorySizeInfo size_info;
    size_info.dims = {image_width, image_height};

    ID3D11Texture2D * buf;
    Status ret = tnn_device->Allocate((void**)&buf, size_info);
    RETURN_VALUE_ON_NEQ(ret, TNN_OK, std::shared_ptr<DirectXMemory>(nullptr));

    DirectXMemory * dx_mem = new DirectXMemory(TNN_DX_TEXTURE);
    dx_mem->SetData(buf, false);
    dx_mem->SetMemoryInfo(data_type, data_format, dims);
    return std::shared_ptr<DirectXMemory>(dx_mem);
}

Status DirectXMemory::Dump() const {

    auto tnn_device = dynamic_cast<DirectXDevice*>(GetDevice(DEVICE_DIRECTX));
    if (!tnn_device) {
        LOGE("Got null directx device");
        return Status(TNNERR_DX_RESOURCE_CREATION);
    }

    auto d3d_device= tnn_device->GetID3DDevice();
    if (!d3d_device) {
        LOGE("Got null d3d device");
        return Status(TNNERR_DX_RESOURCE_CREATION);
    }

    auto d3d_context = tnn_device->GetID3DContext();
    if (!d3d_context) {
        LOGE("Got null d3d context");
        return Status(TNNERR_DX_RESOURCE_CREATION);
    }

    if (TNN_DX_BUFFER == mem_type_) {

        size_t size_in_bytes = DimsVectorUtils::Count(dims_) * DataTypeUtils::GetBytesSize(data_type_);

        d3d_context->Flush();

        D3D11_BUFFER_DESC desc;
        ZeroMemory(&desc, sizeof(desc));

        ID3D11Buffer * src_buffer = (ID3D11Buffer*) data_;
        src_buffer->GetDesc(&desc);
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        desc.Usage = D3D11_USAGE_STAGING;
        desc.BindFlags = 0;
        desc.MiscFlags = 0;

        ID3D11Buffer* debug_buffer;
        HRESULT hr = d3d_device->CreateBuffer(&desc, NULL, &debug_buffer);
        if (FAILED(hr)) {
            LOGE("DirectX create debug Buffer failed ret:0x%X", hr);
            return Status(TNNERR_DX_BUFFER_ALOCATE_ERR, "DirectX create debug Buffer failed.");
        }

        d3d_context->CopyResource(debug_buffer, src_buffer);

        D3D11_MAPPED_SUBRESOURCE mapped_resource;
        hr = d3d_context->Map(debug_buffer, 0, D3D11_MAP_READ, 0, &mapped_resource);
        if (FAILED(hr)) {
            LOGE("DirectX map failed ret:0x%X", hr);
            return Status(TNNERR_DX_MAP_ERR, "DirectX map failed.");
        }

        if (data_type_ != DATA_TYPE_FLOAT) {
            LOGE("only float supports dump now");
            return Status(TNNERR_DX_RESOURCE_CREATION);
        }

        for(int i=0;i<DimsVectorUtils::Count(dims_);i++) {
            printf("dumping [%d]:%.6f\n", i, ((const float *)mapped_resource.pData)[i]);
        }

        d3d_context->Unmap(debug_buffer, 0);
        debug_buffer->Release();

    } else {
        char error_str[128];
        sprintf(error_str, "Dump not support this memory_type");
        return Status(TNNERR_PARAM_ERR, error_str);
    }

    return TNN_OK;
}


std::shared_ptr<DirectXMemoryManager> DirectXMemoryManager::g_singleton_ = nullptr;
std::mutex DirectXMemoryManager::g_mutex_;

DirectXMemoryManager *DirectXMemoryManager::GetInstance() {
    std::unique_lock<std::mutex> lck(g_mutex_);
    if (!g_singleton_) {
        DirectXMemoryManager * mm = new DirectXMemoryManager(); 
        g_singleton_ = std::shared_ptr<DirectXMemoryManager>(mm, [](DirectXMemoryManager *p){delete p;});
    }

    return g_singleton_.get();
}

DirectXMemoryManager::DirectXMemoryManager() {
}

DirectXMemoryManager::~DirectXMemoryManager() {
    memory_map_.clear();
}

Status DirectXMemoryManager::GetRefMemoryFromBlob(Blob * blob, std::shared_ptr<DirectXMemory> & directx_memory) {
    std::unique_lock<std::mutex> lck(g_mutex_);
    auto it = memory_map_.find(blob);

    if (it != memory_map_.end() ) {
        if (it->second->GetData() == blob->GetHandle().base) {
            directx_memory = it->second;
            return TNN_OK;
        }
    }

    auto mem = DirectXMemory::CreateRefMemoryFromBlob(blob);
    if (!mem) {
        LOGE("Creating DirectXMemory from Blob failed\n");
        return Status(TNNERR_DX_RESOURCE_CREATION, "Creating directXMmeory from Blob failed");
    }

    memory_map_.insert(std::make_pair(blob, mem));

    directx_memory = mem;
    return TNN_OK;
}

} // namespace directx

}  // namespace TNN_NS
