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

#ifndef TNN_SOURCE_TNN_DEVICE_RK_NPU_RKNPU_DEVICE_H_
#define TNN_SOURCE_TNN_DEVICE_RK_NPU_RKNPU_DEVICE_H_

#include <cstring>
#include <map>
#include <memory>

#include "tnn/core/abstract_device.h"

namespace TNN_NS {

// @brief RknpuDevice create rknpu memory and rknpu layer acc

class RknpuDevice : public AbstractDevice {
public:
    explicit RknpuDevice(DeviceType device_type);

    ~RknpuDevice();

    virtual BlobMemorySizeInfo Calculate(BlobDesc& desc);

    virtual Status Allocate(void** handle, BlobMemorySizeInfo& size_info);

    virtual Status Allocate(void** handle, MatType mat_type, DimsVector dims);

    virtual Status Free(void* handle);

    virtual Status CopyToDevice(BlobHandle* dst, const BlobHandle* src, BlobDesc& desc, void* command_queue);

    virtual Status CopyFromDevice(BlobHandle* dst, const BlobHandle* src, BlobDesc& desc, void* command_queue);

    virtual AbstractLayerAcc* CreateLayerAcc(LayerType type);

    virtual Context* CreateContext(int device_id);

    static Status RegisterLayerAccCreator(LayerType type, LayerAccCreator* creator);

private:
    static std::map<LayerType, std::shared_ptr<LayerAccCreator>>& GetLayerCreatorMap();
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_RK_NPU_RKNPU_DEVICE_H_
