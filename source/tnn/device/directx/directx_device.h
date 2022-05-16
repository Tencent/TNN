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

#ifndef TNN_SOURCE_TNN_DEVICE_DIRECTX_DIRECTX_DEVICE_H_
#define TNN_SOURCE_TNN_DEVICE_DIRECTX_DIRECTX_DEVICE_H_

#include <map>
#include <memory>
#include <cstring>

#define NOMINMAX
#include <d3dcommon.h>
#include <d3d11.h>
#undef LoadLibrary

#include "tnn/core/abstract_device.h"

namespace TNN_NS {

namespace directx {

// @brief DirectXDevice create memory and layer acc

enum VendorType {
    DX_VENDOR_UNKNOWN = 0,
    DX_VENDOR_NVIDIA,
    DX_VENDOR_INTEL,
    DX_VENDOR_AMD
};

class DirectXDevice : public AbstractDevice {
public:

    explicit DirectXDevice(DeviceType device_type);

    ~DirectXDevice();

    // Initialize the directx device
    Status Init();

    virtual BlobMemorySizeInfo Calculate(BlobDesc& desc);

    virtual Status Allocate(void** handle, BlobMemorySizeInfo& size_info);

    virtual Status Allocate(void** handle, MatType mat_type, DimsVector dims);

    virtual Status Free(void* handle);

    virtual std::shared_ptr<const ImplementedLayout> GetImplementedLayout(LayerType type);

    virtual Status CopyToDevice(BlobHandle* dst, const BlobHandle* src, BlobDesc& desc,
                                void* command_queue);

    virtual Status CopyFromDevice(BlobHandle* dst, const BlobHandle* src, BlobDesc& desc,
                                  void* command_queue);
    
    virtual AbstractLayerAcc *CreateLayerAcc(LayerType type);

    virtual Context *CreateContext(int device_id);

    virtual NetworkType ConvertAutoNetworkType();

    static Status RegisterLayerAccCreator(LayerType type, LayerAccCreator* creator);

    static Status RegisterLayerLayout(LayerType type, std::shared_ptr<ImplementedLayout> layout);

    std::shared_ptr<ID3D11Device>  GetID3DDevice() {
        return device_;
    }

    std::shared_ptr<ID3D11DeviceContext> GetID3DContext() {
        return context_;
    }

    VendorType GetVensorType();

private:
    BlobMemorySizeInfo Calculate1DMemorySize(BlobDesc& desc);
    static std::map<LayerType, std::shared_ptr<LayerAccCreator>> &GetLayerCreatorMap();
    static std::map<LayerType, std::shared_ptr<ImplementedLayout>>& GetLayerLayoutMap();

    std::shared_ptr<ID3D11Device>               device_ = nullptr;
    std::shared_ptr<ID3D11DeviceContext>        context_ = nullptr;

    DXGI_ADAPTER_DESC adapter_desc_;
};

// @brief DirectXLayerAccRegister register DirectXLayerAccCreator
template <typename T>
class DirectXLayerAccRegister {
public:
    explicit DirectXLayerAccRegister(LayerType type) {
        DirectXDevice::RegisterLayerAccCreator(type, new T());
    }
};

class DirectXLayerLayoutRegister {
public:
    explicit DirectXLayerLayoutRegister(LayerType type, std::shared_ptr<ImplementedLayout> layout) {
        DirectXDevice::RegisterLayerLayout(type, layout);
    }
};

} // namespace directx 

} // namespace TNN_NS

#endif // TNN_SOURCE_TNN_DEVICE_DIRECTX_DIRECTX_DEVICE_H_
