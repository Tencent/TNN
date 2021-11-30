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

#ifndef TNN_SOURCE_TNN_DEVICE_METAL_METAL_GATHER_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_METAL_METAL_GATHER_LAYER_ACC_H_

#include "tnn/device/metal/acc/metal_layer_acc.h"

namespace TNN_NS {

// @brief conv layer metal acc
class MetalGatherLayerAcc : public MetalLayerAcc {
public:
    // @brief virtual destrcutor
    virtual ~MetalGatherLayerAcc() {}

    Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    Status AllocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    virtual Status UpdateBlobDataType(const std::vector<Blob *> &inputs,
                                   const std::vector<Blob *> &outputs);

    Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    Status SetKernelEncoderParam(id<MTLComputeCommandEncoder> encoder,
                                    const std::vector<Blob *> &inputs,
                                    const std::vector<Blob *> &outputs);

    Status ComputeThreadSize(const std::vector<Blob *> &inputs,
                                    const std::vector<Blob *> &outputs,
                                    MTLSize &size);

    std::string KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    std::vector<DataFormat> SupportDataFormat(DataType data_type, int dims_size, BlobType blob_type);
protected:
    id<MTLBuffer> buffer_data_;
    id<MTLBuffer> buffer_indices_;
    MTLSize threads_shape_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_METAL_METAL_GATHER_LAYER_ACC_H_
