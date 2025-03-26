// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef TNN_SOURCE_TNN_DEVICE_SNPE_SNPE_DSP_DEVICE_H_
#define TNN_SOURCE_TNN_DEVICE_SNPE_SNPE_DSP_DEVICE_H_

#include "tnn/core/abstract_device.h"

namespace TNN_NS {

// @brief SnpeDspDevice define create memory, context and layer acc interface.
class SnpeDspDevice : public AbstractDevice {
public:
    // @brief constructor
    explicit SnpeDspDevice(DeviceType device_type);

    // @brief virtual destructor
    ~SnpeDspDevice();

    // @brief calculate blob memory size for different dims
    // @param BlobDesc blob description
    // @return blob memory size info
    virtual BlobMemorySizeInfo Calculate(BlobDesc& desc);

    // @brief Allocates mat  memory
    // @param MatType mat type description
    // @param DimsVector mat dims
    // @return blob memory size info
    virtual Status Allocate(void** handle, MatType mat_type, DimsVector dims);

    // @brief Allocates memory
    // @param size info blob size info to allocate
    // @param handle handle blob memory
    // @return TNN_OK if free success, otherwise error code.
    virtual Status Allocate(void** handle, BlobMemorySizeInfo& size_info);

    // @brief Releases memory resources associated by the handle.
    // @return TNN_OK if free success, otherwise error code.
    virtual Status Free(void* handle);

    // @brief Transfer memory from Host to Device
    // @return TNN_OK if copy success, otherwise error code.
    virtual Status CopyToDevice(BlobHandle* dst, const BlobHandle* src, BlobDesc& desc, void* command_queue);

    // @brief Transfer memory from Device to Host
    // @return TNN_OK if copy success, otherwise error code.
    virtual Status CopyFromDevice(BlobHandle* dst, const BlobHandle* src, BlobDesc& desc, void* command_queue);

    // @brief CreateLayerAcc create different layer type acc
    virtual AbstractLayerAcc* CreateLayerAcc(LayerType type);

    // @brief CreateContext create tnn instance device context
    virtual Context* CreateContext(int device_id);
    
    // @brief get implemented layouts on the device by layer type
    //virtual std::shared_ptr<const ImplementedLayout> GetImplementedLayout(LayerType type);

    // @brief auto network type decided by device.
    virtual NetworkType ConvertAutoNetworkType();

private:
    static BlobMemorySizeInfo Calculate1DMemorySize(BlobDesc& desc);
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_SNPE_SNPE_DSP_DEVICE_H_
