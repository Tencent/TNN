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
};

} // namespace directx

}  // namespace TNN_NS
