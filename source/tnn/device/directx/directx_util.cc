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
    return TNN_DX_BUFFER;
    DirectXRuntime* directx_runtime = DirectXRuntime::GetInstance();
    std::vector<size_t> texture_2d_max_size = directx_runtime->GetTexture2DMaxSize();
    ASSERT(texture_2d_max_size.size() == 2);

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
    return TNN_DX_BUFFER;
    if (size_info.dims.size() == 2) {
        TNN_DX_TEXTURE;
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

}
}  // namespace TNN_NS

