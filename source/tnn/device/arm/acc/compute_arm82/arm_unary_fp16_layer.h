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

#ifndef TNN_SOURCE_TNN_DEVICE_ARM_ARM_UNARY_FP16_LAYER_H_
#define TNN_SOURCE_TNN_DEVICE_ARM_ARM_UNARY_FP16_LAYER_H_

#include "tnn/device/arm/acc/Half8.h"
#include "tnn/device/arm/acc/arm_unary_layer_acc.h"
#include "tnn/utils/dims_function_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

#if TNN_ARM82
#define DECLARE_ARM_FP16_UNARY_DO_FORWARD                                                                              \
    virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
#define DEFINE_ARM_FP16_UNARY_LAYER_FUNC(acc_class, op_type)                                                           \
    Status acc_class::ExecFp16(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {                \
        auto input      = inputs[0];                                                                                   \
        auto output     = outputs[0];                                                                                  \
        auto dims       = output->GetBlobDesc().dims;                                                                  \
        auto dim0       = DimsFunctionUtils::GetDim(dims, 0);                                                          \
        auto dim1       = DimsFunctionUtils::GetDim(dims, 1);                                                          \
        int count       = dim0 * ROUND_UP(dim1, 8) * DimsVectorUtils::Count(dims, 2);                                  \
        int count_div8  = UP_DIV(count, 8);                                                                            \
        auto input_ptr  = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(input->GetHandle()));                            \
        auto output_ptr = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(output->GetHandle()));                           \
        OMP_PARALLEL_FOR_                                                                                              \
        for (int n = 0; n < count_div8; n++) {                                                                         \
            Half8::save(output_ptr + n * 8, op_type()(Half8::load(input_ptr + n * 8)));                                \
        }                                                                                                              \
        return TNN_OK;                                                                                                 \
    }
#else
#define DECLARE_ARM_FP16_UNARY_DO_FORWARD
#define DEFINE_ARM_FP16_UNARY_LAYER_FUNC(acc_class, op_type)
#endif  // TNN_ARM82

#define DEFINE_ARM_UNARY_ACC_FP16(type_string, op_type)                                                                \
    class Arm##type_string##LayerAcc : public ArmUnaryLayerAcc {                                                       \
    public:                                                                                                            \
        Arm##type_string##LayerAcc();                                                                                  \
        DECLARE_ARM_FP16_UNARY_DO_FORWARD;                                                                             \
        virtual ~Arm##type_string##LayerAcc(){};                                                                       \
                                                                                                                       \
    private:                                                                                                           \
        DECLARE_ARM_FP16_LAYER_FUNC;                                                                                   \
    };                                                                                                                 \
    DEFINE_ARM_FP16_UNARY_LAYER_FUNC(Arm##type_string##LayerAcc, op_type);

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_ARM_ARM_UNARY_FP16_LAYER_H_
