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

Status DispatchShader(std::shared_ptr<ID3D11ComputeShader> cs, 
                      std::vector<std::shared_ptr<ID3D11ShaderResourceView>> srvs,  
                      std::vector<std::shared_ptr<ID3D11UnorderedAccessView>> uavs,  
                      std::vector<unsigned int> grid) {
    
    auto tnn_device = dynamic_cast<DirectXDevice*>(GetDevice(DEVICE_DIRECTX));
    if (!tnn_device) {
        LOGE("Got null directx device");
        return Status(TNNERR_DX_UNSUPPORTED_DEVICE, "got null directx device");
    }

    auto context = tnn_device->GetID3DContext();

    context->CSSetShader( cs.get(), nullptr, 0 );
    std::vector<ID3D11ShaderResourceView*> srv_ptrs; 
    std::vector<ID3D11UnorderedAccessView*> uav_ptrs; 

    for(auto p : srvs) {srv_ptrs.push_back(p.get());};
    for(auto p : uavs) {uav_ptrs.push_back(p.get());};

    context->CSSetShaderResources( 0, srv_ptrs.size(), srv_ptrs.data());
    context->CSSetUnorderedAccessViews( 0, uav_ptrs.size(), uav_ptrs.data(), nullptr );


    UINT X = grid.size() > 0 ? grid[0] : 1;
    UINT Y = grid.size() > 1 ? grid[1] : 1;
    UINT Z = grid.size() > 2 ? grid[2] : 1;
    context->Dispatch( X, Y, Z );

    context->CSSetShader( nullptr, nullptr, 0 );

    ID3D11UnorderedAccessView* ppUAViewnullptr[1] = { nullptr };
    context->CSSetUnorderedAccessViews( 0, 1, ppUAViewnullptr, nullptr );

    ID3D11ShaderResourceView* ppSRVnullptr[2] = { nullptr, nullptr };
    context->CSSetShaderResources( 0, 2, ppSRVnullptr );

    ID3D11Buffer* ppCBnullptr[1] = { nullptr };
    context->CSSetConstantBuffers( 0, 1, ppCBnullptr ); 

    return TNN_OK;
}

}
}  // namespace TNN_NS

