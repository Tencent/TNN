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

#ifndef TNN_SOURCE_TNN_DEVICE_CPU_CPU_CONV_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_CPU_CPU_CONV_LAYER_ACC_H_

#include <vector>

#include "tnn/core/blob.h"
#include "tnn/device/cpu/acc/cpu_layer_acc.h"
#include "tnn/device/cpu/cpu_device.h"

namespace TNN_NS {

// @brief conv layer cpu acc
class CpuConvLayerAcc : public CpuLayerAcc {
    // @brief virtual destrcutor
    virtual ~CpuConvLayerAcc();

    Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                const std::vector<Blob *> &outputs);

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

private:
    RawBuffer buffer_scale_;

    // @brief buffer_weight_x_bias_ = - zero_point_i * weight + zero_point_i * zero_point_w
    bool do_bias_preprocess_ = false;
    RawBuffer buffer_weight_x_bias_;

    // @brief for conv add fusion
    RawBuffer buffer_add_scale_;
    std::shared_ptr<LayerResource> fp32_resource_ = nullptr;
    // @brief for conv relu6 fusion
    RawBuffer relu6_max_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_CPU_CPU_CONV_LAYER_ACC_H_
