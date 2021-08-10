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

#ifndef TNN_SOURCE_TNN_DEVICE_ARM_ARM_INNER_PRODUCT_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_ARM_ARM_INNER_PRODUCT_LAYER_ACC_H_

#include "tnn/device/arm/acc/arm_layer_acc.h"

namespace TNN_NS {

class ArmInnerProductLayerAcc : public ArmLayerAcc {
public:
    virtual ~ArmInnerProductLayerAcc();

    Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                const std::vector<Blob *> &outputs);

    virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    // alloc for fc weights and pack GOIHW16
    virtual Status allocateBufferWeight(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    // alloc for fc bias and pack c4
    virtual Status allocateBufferBias(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    template <typename T>
    Status Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    template <typename T>
    Status ExecNchw(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

#if TNN_ARM82
    virtual Status allocateBufferWeightHalf(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual Status allocateBufferBiasHalf(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    Status ExecFp16(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    Status ExecNchwFp16(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
#endif  // TNN_ARM82

protected:
    RawBuffer buffer_weight_;
    RawBuffer buffer_bias_;
    RawBuffer buffer_scale_;

    std::function<void(int8_t *, const int8_t *, const int8_t*, const int32_t*,
                       const float*, long, long)> gemv_func_;
    bool support_int8_sdot_ = false;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_ARM_ARM_INNER_PRODUCT_LAYER_ACC_H_
