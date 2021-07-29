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

#ifndef TNN_SOURCE_TNN_DEVICE_ARM_ARM_DEVICE_H_
#define TNN_SOURCE_TNN_DEVICE_ARM_ARM_DEVICE_H_

#include "tnn/core/abstract_device.h"

namespace TNN_NS {

// @brief ArmDevice create cpu memory and cpu layer acc

class ArmDevice : public AbstractDevice {
public:
    explicit ArmDevice(DeviceType device_type);

    virtual ~ArmDevice();

    virtual BlobMemorySizeInfo Calculate(BlobDesc& desc);

    virtual Status Allocate(void** handle, BlobMemorySizeInfo& size_info);

    virtual Status Allocate(BlobHandle* handle, BlobMemorySizeInfo& size_info);

    virtual Status Allocate(void** handle, MatType mat_type, DimsVector dims);

    virtual Status Free(void* handle);

    virtual Status CopyToDevice(BlobHandle* dst, const BlobHandle* src, BlobDesc& desc, void* command_queue);

    virtual Status CopyFromDevice(BlobHandle* dst, const BlobHandle* src, BlobDesc& desc, void* command_queue);

    virtual AbstractLayerAcc* CreateLayerAcc(LayerType type);

    virtual Context* CreateContext(int device_id);

    virtual std::shared_ptr<const ImplementedPrecision> GetImplementedPrecision(LayerType type);

    virtual NetworkType ConvertAutoNetworkType();

    virtual std::shared_ptr<const ImplementedLayout> GetImplementedLayout(LayerType type);

    static Status RegisterLayerAccCreator(LayerType type, LayerAccCreator* creator);

    static Status RegisterLayerPrecision(LayerType type, std::shared_ptr<ImplementedPrecision> precision);

    static Status RegisterLayerLayout(LayerType type, std::shared_ptr<ImplementedLayout> layout);

private:
    BlobMemorySizeInfo Calculate1DMemorySize(BlobDesc& desc);
    static std::map<LayerType, std::shared_ptr<LayerAccCreator>>& GetLayerCreatorMap();
    static std::map<LayerType, std::shared_ptr<ImplementedPrecision>>& GetLayerPrecisionMap();
    static std::map<LayerType, std::shared_ptr<ImplementedLayout>>& GetLayerLayoutMap();
};

//@brief ArmTypeLayerAccRegister register ArmTypeLayerAccCreator
template <typename T>
class ArmTypeLayerAccRegister {
public:
    explicit ArmTypeLayerAccRegister(LayerType type) {
        ArmDevice::RegisterLayerAccCreator(type, new T());
    }
};

class ArmTypeLayerPrecisionRegister {
public:
    explicit ArmTypeLayerPrecisionRegister(LayerType type, std::shared_ptr<ImplementedPrecision> precision) {
        ArmDevice::RegisterLayerPrecision(type, precision);
    }
};

class ArmTypeLayerLayoutRegister {
public:
    explicit ArmTypeLayerLayoutRegister(LayerType type, std::shared_ptr<ImplementedLayout> layout) {
        ArmDevice::RegisterLayerLayout(type, layout);
    }
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_ARM_ARM_DEVICE_FACTORY_H_
