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

#include "tnn/device/directx/directx_device.h"

#include "tnn/core/macro.h"
#include "tnn/device/directx/directx_context.h"
#include "tnn/device/directx/directx_macro.h"
#include "tnn/device/directx/directx_runtime.h"
#include "tnn/device/directx/directx_memory.h"
#include "tnn/device/directx/directx_util.h"
#include "tnn/utils/blob_memory_size_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

namespace directx {

#define LAZY_INIT()                                     \
do {                                                    \
    if (!device_ || !context_)                          \
        RETURN_ON_NEQ(this->Init(), TNN_OK);            \
} while(0)

DirectXDevice::DirectXDevice(DeviceType device_type) : AbstractDevice(device_type) {}

DirectXDevice::~DirectXDevice() {
}


Status DirectXDevice::Init() {
    HRESULT hr = S_OK;
    bool forceRef = false;

    UINT uCreationFlags = D3D11_CREATE_DEVICE_SINGLETHREADED;
#ifdef _DEBUG
    uCreationFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
    D3D_FEATURE_LEVEL flOut;
    static const D3D_FEATURE_LEVEL flvl[] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0 };
    
    bool bNeedRefDevice = false;

    ID3D11Device*               pDevice = nullptr;
    ID3D11DeviceContext*        pContext = nullptr;

    if ( !forceRef )
    {
        hr = D3D11CreateDevice( nullptr,                        // Use default graphics card
                                D3D_DRIVER_TYPE_HARDWARE,    // Try to create a hardware accelerated device
                                nullptr,                        // Do not use external software rasterizer module
                                uCreationFlags,              // Device creation flags
                                flvl,
                                sizeof(flvl) / sizeof(D3D_FEATURE_LEVEL),
                                D3D11_SDK_VERSION,           // SDK version
                                &pDevice,                 // Device out
                                &flOut,                      // Actual feature level created
                                &pContext);              // Context out
        
        if ( SUCCEEDED( hr ) )
        {
            // A hardware accelerated device has been created, so check for Compute Shader support
            // If we have a device >= D3D_FEATURE_LEVEL_11_0 created, full CS5.0 support is guaranteed, no need for further checks
            if ( flOut < D3D_FEATURE_LEVEL_11_0 )            
            {
                // Otherwise, we need further check whether this device support CS4.x (Compute on 10)
                D3D11_FEATURE_DATA_D3D10_X_HARDWARE_OPTIONS hwopts;
                pDevice->CheckFeatureSupport( D3D11_FEATURE_D3D10_X_HARDWARE_OPTIONS, &hwopts, sizeof(hwopts) );
                if ( !hwopts.ComputeShaders_Plus_RawAndStructuredBuffers_Via_Shader_4_x )
                {
                    bNeedRefDevice = true;
                    LOGD("No hardware Compute Shader capable device found, trying to create ref device.");
                }
            }
        }
    }
    
    if ( forceRef || FAILED(hr) || bNeedRefDevice )
    {
        // Either because of failure on creating a hardware device or hardware lacking CS capability, we create a ref device here

        TNN_SAFE_RELEASE( pDevice);
        TNN_SAFE_RELEASE( pContext);
        
        hr = D3D11CreateDevice( nullptr,                        // Use default graphics card
                                D3D_DRIVER_TYPE_REFERENCE,   // Try to create a hardware accelerated device
                                nullptr,                        // Do not use external software rasterizer module
                                uCreationFlags,              // Device creation flags
                                flvl,
                                sizeof(flvl) / sizeof(D3D_FEATURE_LEVEL),
                                D3D11_SDK_VERSION,           // SDK version
                                &pDevice,                 // Device out
                                &flOut,                      // Actual feature level created
                                &pContext);              // Context out
        if ( FAILED(hr) )
        {
            LOGE( "Reference rasterizer device create failure\n" );
            return Status(TNNERR_DX_UNSUPPORTED_DEVICE, "Create directx reference device failed." );
        }
    }

    device_  = std::shared_ptr<ID3D11Device>(pDevice, [](ID3D11Device * p){ TNN_SAFE_RELEASE(p);});
    context_ = std::shared_ptr<ID3D11DeviceContext>(pContext, [](ID3D11DeviceContext * p){ TNN_SAFE_RELEASE(p);});

    return TNN_OK;
}

BlobMemorySizeInfo DirectXDevice::Calculate1DMemorySize(BlobDesc &desc) {
    BlobMemorySizeInfo info;
    info.data_type = desc.data_type;
    int count      = 0;
    if (desc.data_type == DATA_TYPE_INT8) {
        // TODO , check directx int8 data format
        count = desc.dims[0] * ROUND_UP(desc.dims[1], 4) * DimsVectorUtils::Count(desc.dims, 2);
    } else {
        count = DimsVectorUtils::Count(desc.dims);
    }
    info.dims.push_back(count);
    return info;
}

BlobMemorySizeInfo DirectXDevice::Calculate(BlobDesc &desc) {
    DirectXMemoryType mem_type = GetMemoryType(desc);

    if (TNN_DX_BUFFER == mem_type) {
        BlobMemorySizeInfo info = Calculate1DMemorySize(desc);
        desc.data_format = DATA_FORMAT_NCHW;
        return info;
    } else {
        BlobMemorySizeInfo info = Calculate2DCLImageMemorySize(desc);
        return info;
    }
}

Status DirectXDevice::Allocate(void** handle, MatType mat_type, DimsVector dims) {
    LAZY_INIT();

    if (dims.size() != 4) {
        LOGE("invalid dim size: %d\n", (int)dims.size());
        return Status(TNNERR_PARAM_ERR, "invalid dim size");
    }

    BlobDesc desc;
    desc.dims        = dims;
    desc.device_type = GetDeviceType();
    // desc.data_type   = DATA_TYPE_HALF; // try to use half precision
    if (mat_type == N8UC4) {
        auto size_info = Calculate(desc);
        return Allocate(handle, size_info);
    } else {
        LOGE("directx allocator not support this mat type: %d\n", mat_type);
        return Status(TNNERR_PARAM_ERR, "directx not support this mat type");
    }
}

Status DirectXDevice::Allocate(void** handle, BlobMemorySizeInfo& desc) {
    LAZY_INIT();

    DirectXRuntime* directx_runtime = DirectXRuntime::GetInstance();

    if (DATA_TYPE_HALF != desc.data_type && DATA_TYPE_FLOAT != desc.data_type && DATA_TYPE_INT32 != desc.data_type) {
        LOGE("directx allocator not support this data type: %d\n", desc.data_type);
        return Status(TNNERR_PARAM_ERR, "directx not support this data type");
    }

    size_t type_size = 4;
    DXGI_FORMAT format = DXGI_FORMAT_R32G32B32A32_FLOAT;

    if (DATA_TYPE_HALF == desc.data_type) {
        type_size = 2;
        format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    }
    if (DATA_TYPE_INT32 == desc.data_type) {
        type_size = sizeof(int);
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

        ID3D11Texture2D * texture;

        LOGI("DirectX create texture of shape %u x %u\n", desc.dims[0], desc.dims[1] );
        HRESULT hr = device_->CreateTexture2D(&texture_desc, NULL, &texture);
        if (FAILED(hr)) {
            *handle = nullptr;
            LOGE("DirectX create texture failed. erro code %d", (long) hr);
            return Status(TNNERR_DX_TEXTURE_ALOCATE_ERR, "DirectX texture allocation failed.");
        }
        *handle = (void *) texture;

    } else if (TNN_DX_BUFFER == mem_type) {
        // allocate Buffer
        ID3D11Buffer * buffer;

        D3D11_BUFFER_DESC buffer_desc = {};
        buffer_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
        buffer_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
        buffer_desc.Usage = D3D11_USAGE_DEFAULT;
        buffer_desc.ByteWidth = type_size * desc.dims[0]; 

        LOGI("DirectX create buffer of len %u \n", type_size * desc.dims[0]);
        HRESULT hr = device_->CreateBuffer( &buffer_desc, nullptr, &buffer);
        if (FAILED(hr)) {
            *handle = nullptr;
            LOGE("DirectX createbuffer failed. erro code %d", (long) hr);
            return Status(TNNERR_DX_BUFFER_ALOCATE_ERR, "DirectX buffer allocation failed.");
        }
        *handle = (void *)  buffer;
    } else {
        char error_str[128];
        sprintf(error_str, "DirecX not support Allocate (dims=%d)", (int)desc.dims.size());
        return Status(TNNERR_PARAM_ERR, error_str);
    }
    return TNN_OK;
}

Status DirectXDevice::Free(void* handle) {
    if (handle) {
        IUnknown * p = static_cast<IUnknown *>(handle);
        p->Release();
    }
    return TNN_OK;
}

Status DirectXDevice::CopyToDevice(BlobHandle* dst, const BlobHandle* src, BlobDesc& blob_desc, void* command_queue) {
    auto size_info       = Calculate(blob_desc);
    size_t size_in_bytes = GetBlobMemoryBytesSize(size_info);

    LOGD("Copy Data to Device now, size in bytes:%lu shape:%d %d %d %d\n", size_in_bytes,  blob_desc.dims[0], blob_desc.dims[1], blob_desc.dims[2], blob_desc.dims[3]);

    // TODO: Judge blob memory type (texture or buffer ) from blob_desc.format
    // TODO: Add texture to buffer converter
    // Assume all the memory are buffer here.

    context_->Flush();

    ID3D11Buffer * dst_buffer = (ID3D11Buffer*) (dst->base);
    context_->UpdateSubresource(dst_buffer, 0, 0, src->base, 0, 0);

    return TNN_OK;
}

Status DirectXDevice::CopyFromDevice(BlobHandle* dst, const BlobHandle* src, BlobDesc& blob_desc, void* command_queue) {

    auto size_info       = Calculate(blob_desc);
    size_t size_in_bytes = GetBlobMemoryBytesSize(size_info);
    LOGD("Copy Data From Device now, size in bytes:%lu\n", size_in_bytes);

    // TODO: Judge blob memory type (texture or buffer ) from blob_desc.format
    // TODO: Add texture to buffer converter
    // Assume all the memory are buffer here.

    context_->Flush();

    D3D11_BUFFER_DESC desc;
    ZeroMemory(&desc, sizeof(desc));

    ID3D11Buffer * src_buffer = (ID3D11Buffer*) (src->base);
    src_buffer->GetDesc(&desc);
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.MiscFlags = 0;

    ID3D11Buffer* debug_buffer;
    HRESULT hr = device_->CreateBuffer(&desc, NULL, &debug_buffer);
    if (FAILED(hr)) {
        LOGE("DirectX create debug Buffer failed ret:0x%X", hr);
        return Status(TNNERR_DX_BUFFER_ALOCATE_ERR, "DirectX create debug Buffer failed.");
    }

    LOGD("CopyResource 0x%X -> 0x%X\n", src_buffer, debug_buffer);
    context_->CopyResource(debug_buffer, src_buffer);

    D3D11_MAPPED_SUBRESOURCE mapped_resource;
    hr = context_->Map(debug_buffer, 0, D3D11_MAP_READ, 0, &mapped_resource);
    if (FAILED(hr)) {
        LOGE("DirectX map failed ret:0x%X", hr);
        return Status(TNNERR_DX_MAP_ERR, "DirectX map failed.");
    }

    LOGD("memcpy 0x%X -> 0x%X\n", mapped_resource.pData, reinterpret_cast<char*>(src->base) + dst->bytes_offset);
    memcpy(reinterpret_cast<char*>(dst->base) + dst->bytes_offset, 
           mapped_resource.pData, 
           size_in_bytes);

    context_->Unmap(debug_buffer, 0);
    debug_buffer->Release();

    return TNN_OK;
}

AbstractLayerAcc* DirectXDevice::CreateLayerAcc(LayerType type) {
    auto &layer_creator_map = GetLayerCreatorMap();
    if (layer_creator_map.count(type) > 0) {
        return layer_creator_map[type]->CreateLayerAcc(type);
    }
    return NULL;
}

Context* DirectXDevice::CreateContext(int device_id) {
    auto init_fn = [this]() { LAZY_INIT(); return Status(TNN_OK);};
    if (init_fn() != TNN_OK) {
        return nullptr;
    }

    DirectXContext *ctx = new DirectXContext();
    ctx->SetDevice(&device_);
    ctx->SetContext(&context_);

    auto runtime = DirectXRuntime::GetInstance();
    // Caution : very dangerous here, shred_ptr not own the ptr, and ctx might null and crash in later usage .
    // Hack for the blob converter to get the DirectXContext.
    runtime->SetTNNContext(std::shared_ptr<DirectXContext>(ctx, [](DirectXContext * p){}));

    return ctx;
}

NetworkType DirectXDevice::ConvertAutoNetworkType() {
    return NETWORK_TYPE_DEFAULT;
}

Status DirectXDevice::RegisterLayerAccCreator(LayerType type, LayerAccCreator* creator) {
    GetLayerCreatorMap()[type] = std::shared_ptr<LayerAccCreator>(creator);
    return TNN_OK;
}

std::map<LayerType, std::shared_ptr<LayerAccCreator>>& DirectXDevice::GetLayerCreatorMap() {
    static std::map<LayerType, std::shared_ptr<LayerAccCreator>> layer_creator_map;
    return layer_creator_map;
}

std::shared_ptr<const ImplementedLayout> DirectXDevice::GetImplementedLayout(LayerType type) {
    auto &map = GetLayerLayoutMap();
    if (map.find(type) != map.end()) {
        return GetLayerLayoutMap()[type];
    }
    return std::make_shared<ImplementedLayout>();
}

Status DirectXDevice::RegisterLayerLayout(LayerType type, std::shared_ptr<ImplementedLayout> layout) {
    GetLayerLayoutMap()[type] = layout;
    return TNN_OK;
}

std::map<LayerType, std::shared_ptr<ImplementedLayout>> & DirectXDevice::GetLayerLayoutMap() {
    static std::map<LayerType, std::shared_ptr<ImplementedLayout>> layer_layout_map;
    return layer_layout_map;
}

TypeDeviceRegister<DirectXDevice> g_directx_device_register(DEVICE_DIRECTX);

} // namespace directx

} // namespace TNN_NS
