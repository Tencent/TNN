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

#ifndef TNN_SOURCE_TNN_DEVICE_ARM_ARM_CONV_INT8_LAYER_ACC_INDIRECT_H_
#define TNN_SOURCE_TNN_DEVICE_ARM_ARM_CONV_INT8_LAYER_ACC_INDIRECT_H_

#include "tnn/device/arm/acc/convolution/arm_conv_int8_layer_common.h"

namespace TNN_NS {

// @brief conv layer cpu acc

class ArmConvInt8LayerIndirect : public ArmConvInt8LayerCommon {
public:
    virtual ~ArmConvInt8LayerIndirect();

    Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                const std::vector<Blob *> &outputs);

    virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    static bool isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                           const std::vector<Blob *> &outputs);

    virtual Status allocateBufferWeightBias(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual Status allocateBufferInput(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
private:
    Status initIndirectionBuffer(const std::vector<Blob *> &inputs, const std::vector<Blob *> & outputs,
                                size_t output_tile_size, size_t tiled_output_size);
    RawBuffer indirection_buffer_;
    RawBuffer zero_buffer_;
    int       zero_buffer_offset_;
    bool      blob_allocated_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_ARM_ARM_CONV_INT8_LAYER_ACC_INDIRECT_H_
