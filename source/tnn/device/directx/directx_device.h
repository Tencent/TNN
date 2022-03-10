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

#include <d3dcommon.h>
#include <d3d11.h>
#undef min
#undef max

#include "tnn/core/abstract_device.h"

namespace TNN_NS {

// @brief DirectXDevice create memory and layer acc

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

private:
    BlobMemorySizeInfo Calculate1DMemorySize(BlobDesc& desc);
    static std::map<LayerType, std::shared_ptr<LayerAccCreator>> &GetLayerCreatorMap();

    std::shared_ptr<ID3D11Device>               device_ = nullptr;
    std::shared_ptr<ID3D11DeviceContext>        context_ = nullptr;

};

// @brief DirectXTypeLayerAccRegister register DirectXTypeLayerAccCreator
template <typename T>
class DirectXTypeLayerAccRegister {
public:
    explicit DirectXTypeLayerAccRegister(LayerType type) {
        DirectXDevice::RegisterLayerAccCreator(type, new T());
    }
};

} // namespace TNN_NS

#endif // TNN_SOURCE_TNN_DEVICE_DIRECTX_DIRECTX_DEVICE_H_
