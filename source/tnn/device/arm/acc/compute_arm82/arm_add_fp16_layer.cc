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

#include "tnn/device/arm/acc/arm_add_layer_acc.h"

#include "tnn/utils/dims_vector_utils.h"
#include "tnn/device/arm/acc/Half8.h"

namespace TNN_NS {

#if TNN_ARM82
void _operator_add_fp16(fp16_t *output_ptr, fp16_t *input0_ptr, fp16_t *input1_ptr, DimsVector &dims0,
                                  DimsVector &dims1) {
    DimsVector dims = DimsVectorUtils::Max(dims0, dims1);
    AddOpType type = ADD_ELEMENT;
    auto _input0   = input0_ptr;
    auto _input1   = input1_ptr;
    OperatorAddPreparation();

    int count      = ROUND_UP(dims[1], 8) * dims[2] * dims[3];
    int count_div8 = UP_DIV(count, 8);

    if (type == ADD_SINGLE) {
        // broadcast single
        count_div8 *= dims[0];
        for (int n = 0; n < count_div8; n++) {
            Half8::save(output_ptr + n * 8, Half8::load(_input0 + n * 8) + Half8(_input1[0]));
        }
    } else if (type == ADD_ELEMENT) {
        // no broadcast
        if (dims0[0] == dims1[0] && dims0[1] == dims1[1]) {
            count_div8 *= dims[0];
            for (int n = 0; n < count_div8; n++) {
                Half8::save(output_ptr + n * 8, Half8::load(_input0 + n * 8) + Half8::load(_input1 + n * 8));
            }
        } else if (dims0[1] == dims1[1]) {
            // broadcast chw
            for (int batch = 0; batch < dims[0]; batch++) {
                auto input0_batch_ = _input0 + count * batch;
                auto output_batch_ = output_ptr + count * batch;
                for (int n = 0; n < count_div8; n++) {
                    Half8::save(output_batch_ + n * 8,
                                Half8::load(input0_batch_ + n * 8) + Half8::load(_input1 + n * 8));
                }
            }
        } else {
            // broadcast hw
            for (int batch = 0; batch < dims[0]; batch++) {
                auto input0_batch_ = _input0 + count * batch;
                auto output_batch_ = output_ptr + count * batch;
                for (int n = 0; n < count_div8; n++) {
                    auto hw_index = n % (dims[2] * dims[3]);
                    Half8::save(output_batch_ + n * 8,
                                Half8::load(input0_batch_ + n * 8) + Half8(_input1[hw_index * 8]));
                }
            }
        }
    } else if (type == ADD_CHANNEL) {
        // broadcast channel
        count_div8 *= dims[0];
        for (int n = 0; n < count_div8; n++) {
            int b               = n / (dims[2] * dims[3] * UP_DIV(dims[1], 8));
            int channel_8_index = n / (dims[2] * dims[3]) - b * UP_DIV(dims[1], 8);
            Half8::save(output_ptr + n * 8,
                        Half8::load(_input0 + n * 8) + Half8::load(_input1 + channel_8_index * 8));
        }
    } else {
        LOGE("Error: invalid add type\n");
    }
}
#endif

}  // namespace TNN_NS
