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
#include <unordered_map>
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
#include "tnn/utils/dims_utils.h"

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


    static std::unordered_map<std::string, std::shared_ptr<ID3D11ComputeShader>> g_shader_map;
    static std::mutex g_mutex;

    std::unique_lock<std::mutex> lck(g_mutex);
    auto it = g_shader_map.find(kernel_name);
    if (it != g_shader_map.end()) {
        shader = it->second;
        return TNN_OK;
    }

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

    g_shader_map.insert(std::make_pair(kernel_name, shader));

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

Status UpdateTexture2D(void* data_ptr,
                       std::vector<int> dims,
                       std::shared_ptr<DirectXMemory> &texture_memory) {

    auto tnn_device = dynamic_cast<DirectXDevice*>(GetDevice(DEVICE_DIRECTX));
    if (!tnn_device) {
        LOGE("Got null directx device");
        return Status(TNNERR_CONTEXT_ERR, "got null directx device");
    }

    auto device = tnn_device->GetID3DDevice();
    if (!device) {
        LOGE("Got null ID3Ddevice");
        return  Status(TNNERR_CONTEXT_ERR, "got null directx device");
    }

    shared_ptr<DirectXMemory> buffer = DirectXMemory::CreateBufferMemoryFromHost(
        data_ptr, dims, DATA_TYPE_FLOAT, DATA_FORMAT_NCHW);
    if (!buffer) {
        LOGE("param transfer to GPU failed.");
        return Status(TNNERR_DX_BUFFER_ALOCATE_ERR, "param transfer to GPU failed.");
    }

    auto buffer_srv = buffer->GetSRV();
    auto texture_uav = texture_memory->GetUAV();

    ParamCB param_cb_host = {1, 1, 1, 1,
                             0, 0, 0, 0,
                             DimsFunctionUtils::GetDim(dims, 0), DimsFunctionUtils::GetDim(dims, 1),
                             DimsFunctionUtils::GetDim(dims, 2), DimsFunctionUtils::GetDim(dims, 3)};

    std::shared_ptr<ID3D11Buffer> param_cb;
    Status status = CreateConstBuffer<ParamCB>(param_cb_host, device, param_cb);
    RETURN_ON_NEQ(status, TNN_OK);

    LOGD("kernel name: NCHWToNHC4W4\n");
    std::shared_ptr<ID3D11ComputeShader> cs;
    status = GetShaderByName("NCHWToNHC4W4", cs);
    RETURN_ON_NEQ(status, TNN_OK);

    const int THREADS_PER_BLOCK = 128;
    const int ELE_PER_THREAD    = 4;

    int batch, channel, height, width;
    batch            = DimsFunctionUtils::GetDim(dims, 0);
    channel          = DimsFunctionUtils::GetDim(dims, 1);
    height           = DimsFunctionUtils::GetDim(dims, 2);
    width            = DimsFunctionUtils::GetDim(dims, 3);
    int image_width  = UP_DIV(channel, 4) * width;
    int image_height = batch * height;
    Status  ret = DispatchShader(cs, {buffer_srv}, {texture_uav}, {param_cb.get()}, {image_width,image_height,1});

    return  ret;
}

Status UpdateConv2DFilterTexture2D(void* data_ptr,
                       std::vector<int> dims,
                       int image_width,
                       int image_height,
                       std::shared_ptr<DirectXMemory> &texture_memory) {

    auto tnn_device = dynamic_cast<DirectXDevice*>(GetDevice(DEVICE_DIRECTX));
    if (!tnn_device) {
        LOGE("Got null directx device");
        return Status(TNNERR_CONTEXT_ERR, "got null directx device");
    }

    auto device = tnn_device->GetID3DDevice();
    if (!device) {
        LOGE("Got null ID3Ddevice");
        return  Status(TNNERR_CONTEXT_ERR, "got null directx device");
    }

    shared_ptr<DirectXMemory> buffer = DirectXMemory::CreateBufferMemoryFromHost(
        data_ptr, dims, DATA_TYPE_FLOAT, DATA_FORMAT_NCHW);
    if (!buffer) {
        LOGE("param transfer to GPU failed.");
        return Status(TNNERR_DX_BUFFER_ALOCATE_ERR, "param transfer to GPU failed.");
    }

    auto buffer_srv = buffer->GetSRV();
    auto texture_uav = texture_memory->GetUAV();

    typedef struct launch_param {
        DirectX::XMUINT4 shape;
    } launch_param_t;

    launch_param_t args;
    args.shape  = DirectX::XMUINT4(DimsFunctionUtils::GetDim(dims, 0), DimsFunctionUtils::GetDim(dims, 1),
                                   DimsFunctionUtils::GetDim(dims, 2), DimsFunctionUtils::GetDim(dims, 3));

    std::shared_ptr<ID3D11Buffer> param_cb;
    Status status = CreateConstBuffer<launch_param_t>(args, device, param_cb);
    RETURN_ON_NEQ(status, TNN_OK);

    LOGD("kernel name: Conv2DFilterToNHC4W4\n");
    std::shared_ptr<ID3D11ComputeShader> cs;
    status = GetShaderByName("Conv2DFilterToNHC4W4", cs);
    RETURN_ON_NEQ(status, TNN_OK);

    Status  ret = DispatchShader(cs, {buffer_srv}, {texture_uav}, {param_cb.get()}, {image_width,image_height,1});

    return  ret;
}

Status UpdateConvDWFilterTexture2D(void* data_ptr,
                                   std::vector<int> dims,
                                   int image_width,
                                   int image_height,
                                   std::shared_ptr<DirectXMemory> &texture_memory) {

    auto tnn_device = dynamic_cast<DirectXDevice*>(GetDevice(DEVICE_DIRECTX));
    if (!tnn_device) {
        LOGE("Got null directx device");
        return Status(TNNERR_CONTEXT_ERR, "got null directx device");
    }

    auto device = tnn_device->GetID3DDevice();
    if (!device) {
        LOGE("Got null ID3Ddevice");
        return  Status(TNNERR_CONTEXT_ERR, "got null directx device");
    }

    shared_ptr<DirectXMemory> buffer = DirectXMemory::CreateBufferMemoryFromHost(
        data_ptr, dims, DATA_TYPE_FLOAT, DATA_FORMAT_NCHW);
    if (!buffer) {
        LOGE("param transfer to GPU failed.");
        return Status(TNNERR_DX_BUFFER_ALOCATE_ERR, "param transfer to GPU failed.");
    }

    auto buffer_srv = buffer->GetSRV();
    auto texture_uav = texture_memory->GetUAV();

    typedef struct launch_param {
        DirectX::XMUINT4 shape;
    } launch_param_t;

    launch_param_t args;
    args.shape  = DirectX::XMUINT4(DimsFunctionUtils::GetDim(dims, 0), DimsFunctionUtils::GetDim(dims, 1),
                                   DimsFunctionUtils::GetDim(dims, 2), DimsFunctionUtils::GetDim(dims, 3));

    std::shared_ptr<ID3D11Buffer> param_cb;
    Status status = CreateConstBuffer<launch_param_t>(args, device, param_cb);
    RETURN_ON_NEQ(status, TNN_OK);

    LOGD("kernel name: ConvDWFilterToNHC4W4\n");
    std::shared_ptr<ID3D11ComputeShader> cs;
    status = GetShaderByName("ConvDWFilterToNHC4W4", cs);
    RETURN_ON_NEQ(status, TNN_OK);

    Status  ret = DispatchShader(cs, {buffer_srv}, {texture_uav}, {param_cb.get()}, {image_width,image_height,1});

    return  ret;
}

}  // namespace directx
}  // namespace TNN_NS

