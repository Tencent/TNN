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

#ifndef TNN_SOURCE_TNN_DEVICE_ARM_ARM_SOFTMAX_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_ARM_ARM_SOFTMAX_LAYER_ACC_H_

#include "tnn/device/arm/acc/arm_layer_acc.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

DECLARE_ARM_ACC(Softmax, LAYER_SOFTMAX);

#define SoftmaxPreparation()                                                                                           \
    auto in_data_type = inputs[0]->GetBlobDesc().data_type;                                                            \
    auto axis         = layer_param->axis;                                                                             \
    auto input        = inputs[0];                                                                                     \
    auto output       = outputs[0];                                                                                    \
    auto dims         = output->GetBlobDesc().dims;                                                                    \
    auto hw           = DimsVectorUtils::Count(dims, 2);                                                               \
    auto batch        = dims[0];                                                                                       \
    bool packed       = input->GetBlobDesc().data_format != DATA_FORMAT_NCHW;                                          \
    size_t count      = hw * batch * ROUND_UP(dims[1], packed ? 4 : 1);                                                \
    int inside        = 1;                                                                                             \
    int outside       = 1;                                                                                             \
    int channel       = 1;                                                                                             \
    for (int i = 1; i < axis; i++) {                                                                                   \
        outside *= dims[i];                                                                                            \
    }                                                                                                                  \
    channel = dims[axis];                                                                                              \
    for (int i = axis + 1; i < dims.size(); i++) {                                                                     \
        inside *= dims[i];                                                                                             \
    }                                                                                                                  \
    auto step_y = channel * inside;

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_ARM_ARM_SOFTMAX_LAYER_ACC_H_
