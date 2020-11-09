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

#ifndef TNN_SOURCE_TNN_CORE_ABSTRACT_DEVICE_H_
#define TNN_SOURCE_TNN_CORE_ABSTRACT_DEVICE_H_

#include "tnn/core/abstract_layer_acc.h"
#include "tnn/core/blob.h"
#include "tnn/core/common.h"
#include "tnn/core/context.h"
#include "tnn/core/layer_type.h"
#include "tnn/core/status.h"
#include "tnn/memory_manager/blob_memory_size_info.h"
#include "tnn/utils/blob_converter.h"

namespace TNN_NS {

struct ImplementedPrecision {
    bool fp32_implemented  = false;
    bool fp16_implemented  = false;
    bool bfp16_implemented = false;
};

// @brief AbstractDevice define create memory, context and layer acc interface.
class AbstractDevice {
public:
    // @brief constructor
    explicit AbstractDevice(DeviceType);

    // @brief virtual destructor
    virtual ~AbstractDevice();

    // @brief calculate blob memory size for different dims
    // @param BlobDesc blob description
    // @return blob memory size info
    virtual BlobMemorySizeInfo Calculate(BlobDesc& desc) = 0;

    // @brief Allocates mat  memory
    // @param MatType mat type description
    // @param DimsVector mat dims
    // @return blob memory size info
    virtual Status Allocate(void** handle, MatType mat_type, DimsVector dims) = 0;

    // @brief Allocates memory
    // @param size info blob size info to allocate
    // @param handle handle blob memory
    // @return TNN_OK if free success, otherwise error code.
    virtual Status Allocate(void** handle, BlobMemorySizeInfo& size_info) = 0;

    // @brief Releases memory resources associated by the handle.
    // @return TNN_OK if free success, otherwise error code.
    virtual Status Free(void* handle) = 0;

    // @brief Transfer memory from Host to Device
    // @return TNN_OK if copy success, otherwise error code.
    virtual Status CopyToDevice(BlobHandle* dst, const BlobHandle* src, BlobDesc& desc, void* command_queue) = 0;

    // @brief Transfer memory from Device to Host
    // @return TNN_OK if copy success, otherwise error code.
    virtual Status CopyFromDevice(BlobHandle* dst, const BlobHandle* src, BlobDesc& desc, void* command_queue) = 0;

    // @brief CreateLayerAcc create different layer type acc
    virtual AbstractLayerAcc* CreateLayerAcc(LayerType type) = 0;

    // @brief CreateContext create tnn instance device context
    virtual Context* CreateContext(int device_id) = 0;

    // @brief get implemented precisions on the device by layer type
    virtual std::shared_ptr<const ImplementedPrecision> GetImplementedPrecision(LayerType type);

    // @brief get factory device type
    DeviceType GetDeviceType();

private:
    DeviceType device_type_;
};

// @brief GetGlobalDeviceMap device type map
std::map<DeviceType, std::shared_ptr<AbstractDevice>>& GetGlobalDeviceMap();

// @brief Get Device
AbstractDevice* GetDevice(DeviceType type);

// @brief TypeDeviceRegister contruct register device
template <typename T>
class TypeDeviceRegister {
public:
    explicit TypeDeviceRegister(DeviceType type) {
        auto &device_map = GetGlobalDeviceMap();
        if (device_map.find(type) == device_map.end()) {
            device_map[type] = std::shared_ptr<T>(new T(type));
        }
    }
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_CORE_ABSTRACT_DEVICE_H_
