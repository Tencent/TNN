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

#include "tnn/device/directx/directx_util.h"

#include <string.h>
#include <cstdlib>
#if TNN_PROFILE
#include <chrono>
#endif

#include "tnn/core/blob.h"
#include "tnn/core/macro.h"

#include "tnn/device/directx/directx_device.h"
#include "tnn/device/directx/directx_memory.h"
#include "tnn/device/directx/directx_runtime.h"
#include "tnn/memory_manager/blob_memory_size_info.h"
#include "tnn/utils/blob_memory_size_utils.h"

#include "tnn/device/directx/directx_kernels.h"

namespace TNN_NS {
namespace directx {

// Tell the memory type from a blob Description sruct
DirectXMemoryType GetMemoryType(BlobDesc desc) {
//    return TNN_DX_BUFFER;
    DirectXRuntime* directx_runtime = DirectXRuntime::GetInstance();
    std::vector<size_t> texture_2d_max_size = directx_runtime->GetTexture2DMaxSize();
    ASSERT(texture_2d_max_size.size() == 2);

    desc.data_format = DATA_FORMAT_NHC4W4;
    BlobMemorySizeInfo info = Calculate2DCLImageMemorySize(desc);
    ASSERT(info.dims.size() == 2);

    if (info.dims[0] > texture_2d_max_size[0] || info.dims[1] > texture_2d_max_size[1]) {
        LOGD("Exceed DirectX Texture2D limit, dims: [%d, %d]\n", info.dims[0], info.dims[1]);
        return TNN_DX_BUFFER;
    }
    return TNN_DX_TEXTURE;
}

// Tell the memory type from a blob memory size info, which is used by the AbstactDevice::Allocate function
DirectXMemoryType GetMemoryType(BlobMemorySizeInfo size_info) {
//    return TNN_DX_BUFFER;
    if (size_info.dims.size() == 2) {
        return TNN_DX_TEXTURE;
    }
    return TNN_DX_BUFFER;
}

Status DispatchShader(const std::shared_ptr<ID3D11ComputeShader> cs, 
                      const std::vector<std::shared_ptr<ID3D11ShaderResourceView>> srvs,  
                      const std::vector<std::shared_ptr<ID3D11UnorderedAccessView>> uavs,  
                      const std::vector<ID3D11Buffer*> constant_buffers,
                      const std::vector<int> grid) {
    
    auto tnn_device = dynamic_cast<DirectXDevice*>(GetDevice(DEVICE_DIRECTX));
    if (!tnn_device) {
        LOGE("Got null directx device");
        return Status(TNNERR_DX_UNSUPPORTED_DEVICE, "got null directx device");
    }

    auto context = tnn_device->GetID3DContext();
    if (!context) {
        LOGE("Got null d3d device");
        return Status(TNNERR_DX_UNSUPPORTED_DEVICE, "got null d3d context");
    }

    context->CSSetShader( cs.get(), nullptr, 0 );
    std::vector<ID3D11ShaderResourceView*> srv_ptrs;
    std::vector<ID3D11UnorderedAccessView*> uav_ptrs;

    for(auto p : srvs) {srv_ptrs.push_back(p.get());};
    for(auto p : uavs) {uav_ptrs.push_back(p.get());};

    context->CSSetShaderResources( 0, srv_ptrs.size(), srv_ptrs.data());
    context->CSSetUnorderedAccessViews( 0, uav_ptrs.size(), uav_ptrs.data(), nullptr );

    if (constant_buffers.size() > 0) {
        context->CSSetConstantBuffers(0, constant_buffers.size(), constant_buffers.data());
    }

    UINT X = grid.size() > 0 ? grid[0] : 1;
    UINT Y = grid.size() > 1 ? grid[1] : 1;
    UINT Z = grid.size() > 2 ? grid[2] : 1;
    context->Dispatch( X, Y, Z );

    context->CSSetShader( nullptr, nullptr, 0 );

    for(size_t i=0;i<srv_ptrs.size();i++) { srv_ptrs[i] = nullptr; }
    for(size_t i=0;i<uav_ptrs.size();i++) { uav_ptrs[i] = nullptr; }

    context->CSSetShaderResources( 0, srv_ptrs.size(), srv_ptrs.data());
    context->CSSetUnorderedAccessViews( 0, uav_ptrs.size(), uav_ptrs.data(), nullptr );

    if (constant_buffers.size() > 0) {
        std::vector<ID3D11Buffer *> null_cbs(constant_buffers.size(), nullptr);
        context->CSSetConstantBuffers(0, null_cbs.size(), null_cbs.data());
    }

    return TNN_OK;
}

Status GetShaderByName(const std::string kernel_name, std::shared_ptr<ID3D11ComputeShader> &shader ) {



    auto tnn_device = dynamic_cast<DirectXDevice*>(GetDevice(DEVICE_DIRECTX));
    if (!tnn_device) {
        LOGE("Got null directx device");
        return Status(TNNERR_DX_UNSUPPORTED_DEVICE, "got null directx device");
    }

    auto device= tnn_device->GetID3DDevice();
    if (!device) {
        LOGE("Got null d3d device");
        return Status(TNNERR_DX_UNSUPPORTED_DEVICE, "got null d3d device");
    }

    auto kernel_map = get_kernel_map();
    auto kernel_size_map = get_kernel_size_map();

    // LOGI("kenrel %s len:%lu lastbyte:<%c>\n", kernel_name.c_str(), kernel_size_map[kernel_name], kernel_map[kernel_name][kernel_size_map[kernel_name] - 1]);
    // const BYTE * ptr = kernel_map[kernel_name];
    // for(int i=0;i<kernel_size_map[kernel_name];i++)
    // {
    //     printf("%c", ptr[i]);
    // }
    // printf("\n");

    ID3D11ComputeShader * p_shader;
    // Create the Matrix Transpose Compute Shader
    HRESULT hr = device->CreateComputeShader(kernel_map[kernel_name], kernel_size_map[kernel_name], nullptr, &p_shader);
    if( FAILED( hr ) ) {
        LOGE("create compute shader failed");
        return Status(TNNERR_DX_SHADER_CREATE_ERR, "create shader failed");
    }

    shader = std::shared_ptr<ID3D11ComputeShader>(p_shader, [](ID3D11ComputeShader * p) {p->Release();} );

    return TNN_OK;
}

Status GetID3DDevice(std::shared_ptr<ID3D11Device> &device) {

    auto tnn_device = dynamic_cast<DirectXDevice*>(GetDevice(DEVICE_DIRECTX));
    if (!tnn_device) {
        LOGE("Got null directx device");
        return Status(TNNERR_DX_UNSUPPORTED_DEVICE, "got null directx device");
    }

    auto d3d_device= tnn_device->GetID3DDevice();
    if (!d3d_device) {
        LOGE("Got null d3d device");
        return Status(TNNERR_DX_UNSUPPORTED_DEVICE, "got null d3d device");
    }

    device = d3d_device;
    return TNN_OK;
}

Status GetID3DContext(std::shared_ptr<ID3D11DeviceContext> &context) {
    auto tnn_device = dynamic_cast<DirectXDevice*>(GetDevice(DEVICE_DIRECTX));
    if (!tnn_device) {
        LOGE("Got null directx device");
        return Status(TNNERR_DX_UNSUPPORTED_DEVICE, "got null directx device");
    }

    auto d3d_context = tnn_device->GetID3DContext();
    if (!d3d_context) {
        LOGE("Got null d3d context");
        return Status(TNNERR_DX_UNSUPPORTED_DEVICE, "got null d3d context");
    }

    context = d3d_context;
    return TNN_OK;
}


Status AllocateBuffer(std::shared_ptr<DirectXMemory> buffer_out,
                      BlobMemorySizeInfo& desc,
                      const void * inital_data){

    auto tnn_device = dynamic_cast<DirectXDevice*>(GetDevice(DEVICE_DIRECTX));
    if (!tnn_device) {
        LOGE("Got null directx device");
        return Status(TNNERR_DX_UNSUPPORTED_DEVICE, "got null directx device");
    }

    auto device = tnn_device->GetID3DDevice();
    ID3D11Device* pDevice = device.get();

    if (DATA_TYPE_HALF != desc.data_type && DATA_TYPE_FLOAT != desc.data_type && DATA_TYPE_INT32 != desc.data_type && DATA_TYPE_INT8 != desc.data_type) {
        LOGE("directx allocator not support this data type: %d\n", desc.data_type);
        return Status(TNNERR_PARAM_ERR, "directx not support this data type");
    }

    size_t type_size = sizeof(float);
    DXGI_FORMAT format = DXGI_FORMAT_R32G32B32A32_FLOAT;

    if (DATA_TYPE_HALF == desc.data_type) {
        type_size = 2;
        format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    }
    if (DATA_TYPE_INT32 == desc.data_type) {
        type_size = sizeof(int);
        format = DXGI_FORMAT_R8G8B8A8_UINT;
    }
    if (DATA_TYPE_INT8 == desc.data_type) {
        type_size = sizeof(uint8_t);
        format = DXGI_FORMAT_R8G8B8A8_UINT;
    }

    DirectXMemoryType mem_type = GetMemoryType(desc);

    if (TNN_DX_TEXTURE == mem_type) {
        D3D11_TEXTURE2D_DESC texture_desc;
        ZeroMemory(&texture_desc, sizeof(texture_desc));
        texture_desc.Width = (UINT)(desc.dims[0]);
        texture_desc.Height = (UINT)(desc.dims[1]);
        texture_desc.MipLevels = 1;
        texture_desc.Format = format;
        texture_desc.Usage = D3D11_USAGE_DEFAULT;
        texture_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        texture_desc.CPUAccessFlags = 0;
        texture_desc.MiscFlags = 0;

        D3D11_SUBRESOURCE_DATA srd = {};
        srd.pSysMem = inital_data;
        srd.SysMemPitch = 0;
        srd.SysMemSlicePitch = 0;

        ID3D11Texture2D * texture;

        LOGI("DirectX create texture of shape %u x %u\n", desc.dims[0], desc.dims[1] );
        HRESULT hr = pDevice->CreateTexture2D(&texture_desc, &srd, &texture);
        if (FAILED(hr)) {
            buffer_out->SetData(nullptr, false);
            LOGE("DirectX create texture failed. erro code %d", (long) hr);
            return Status(TNNERR_DX_TEXTURE_ALOCATE_ERR, "DirectX texture allocation failed.");
        }
        buffer_out->SetData(texture, false);

    } else if (TNN_DX_BUFFER == mem_type) {
        // allocate Buffer
        ID3D11Buffer * buffer;

        D3D11_BUFFER_DESC buffer_desc = {};
        buffer_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
        buffer_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
        buffer_desc.Usage = D3D11_USAGE_DEFAULT;
        buffer_desc.ByteWidth = type_size * desc.dims[0];

        D3D11_SUBRESOURCE_DATA srd = {};
        srd.pSysMem = inital_data;
        srd.SysMemPitch = 0;
        srd.SysMemSlicePitch = 0;

        LOGI("DirectX create buffer of len %u \n", type_size * desc.dims[0]);
        HRESULT hr = pDevice->CreateBuffer( &buffer_desc, &srd, &buffer);
        if (FAILED(hr)) {
            buffer_out->SetData(nullptr, false);
            LOGE("DirectX createbuffer failed. erro code %d", (long) hr);
            return Status(TNNERR_DX_BUFFER_ALOCATE_ERR, "DirectX buffer allocation failed.");
        }
        buffer_out->SetData(buffer, false);

    } else {
        char error_str[128];
        sprintf(error_str, "DirecX not support Allocate (dims=%d)", (int)desc.dims.size());
        return Status(TNNERR_PARAM_ERR, error_str);
    }

    return TNN_OK;
}

Status AllocateConstantBuffer(ID3D11Buffer* &pInputCBBuffer,
                              ParamCB &paramCB_data){

    auto tnn_device = dynamic_cast<DirectXDevice*>(GetDevice(DEVICE_DIRECTX));
    if (!tnn_device) {
        LOGE("Got null directx device");
        return Status(TNNERR_DX_UNSUPPORTED_DEVICE, "got null directx device");
    }

    auto device = tnn_device->GetID3DDevice();
    ID3D11Device* pDevice = device.get();

    D3D11_BUFFER_DESC desc = {};
    desc.ByteWidth = sizeof(ParamCB);
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    desc.CPUAccessFlags = 0u;
    desc.StructureByteStride = 0u;
    desc.MiscFlags = 0u;

    D3D11_SUBRESOURCE_DATA srd = {};
    srd.pSysMem = &paramCB_data;
    srd.SysMemPitch = 0;
    srd.SysMemSlicePitch = 0;

    HRESULT hr = pDevice->CreateBuffer(&desc, &srd, &pInputCBBuffer);
    if (FAILED(hr)) {
        LOGE("DirectX create constant buffer failed. erro code %d", (long) hr);
        return Status(TNNERR_DX_BUFFER_ALOCATE_ERR, "DirectX constant buffer allocation failed.");
    }

    return TNN_OK;
}

}  // namespace directx
}  // namespace TNN_NS

