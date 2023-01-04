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

#include "tnn/core/macro.h"
#include "tnn/device/arm/acc/Half8.h"
#include "tnn/device/arm/acc/arm_glu_layer_acc.h"
#include "tnn/device/arm/acc/arm_layer_acc.h"
#include "tnn/utils/dims_function_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

#if TNN_ARM82

Status ArmGLULayerAcc::ExecFp16(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto &input_blob       = inputs[0];
    auto &output_blob      = outputs[0];
    const auto &input_dims = input_blob->GetBlobDesc().dims;
    const auto &param      = dynamic_cast<GLULayerParam *>(param_);
    const int axis         = param->axis;
    auto *input_ptr        = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(input_blob->GetHandle()));
    auto *output_ptr       = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(output_blob->GetHandle()));

    bool is_split_channel = false;
    if (axis == 1) {
        is_split_channel = true;
        if (k_param_->ic_r8 % 16 != 0) {
            LOGE("ArmGLULayerAcc does not support for now\n");
            return {TNNERR_UNSUPPORT_NET, "ArmGLULayerAcc does not support for now\n"};
        }
    }
    if (is_split_channel) {
        const int batch = input_dims[0];
        const int ic_r8 = k_param_->ic_r8;
        const int oc_r8 = k_param_->oc_r8;
        const int count = DimsVectorUtils::Count(input_dims, axis + 1);
        Half8 one_v     = Half8((fp16_t)1.0f);
        for (int b = 0; b < batch; ++b) {
            auto *input_batch  = input_ptr + b * ic_r8 * count;
            auto *output_batch = output_ptr + b * oc_r8 * count;
            for (int c = 0; c < oc_r8; c += 8) {
                auto *first_slice_ptr  = input_batch + c * count;
                auto *second_slice_ptr = input_batch + (c + oc_r8) * count;
                auto *output_slice_ptr = output_batch + c * count;
                for (int i = 0; i < count; ++i) {
                    Half8 b_v0 = Half8::load(second_slice_ptr + i * 8);
                    b_v0       = Half8::neg(b_v0);
                    b_v0       = Half8::exp(b_v0);
                    Half8 a_v0 = Half8::load(first_slice_ptr + i * 8);
                    b_v0       = b_v0 + one_v;
                    Half8 o_v0 = Half8::div(a_v0, b_v0);
                    Half8::save(output_slice_ptr + i * 8, o_v0);
                }
            }
        }
    } else {
        Half8 one_v          = Half8((fp16_t)1.0f);
        const int batch      = input_dims[0] * k_param_->ic_r8 * DimsVectorUtils::Count(input_dims, 2, axis);
        const int split_dim  = input_dims[axis];
        const int count      = DimsVectorUtils::Count(input_dims, axis + 1);
        const int output_dim = split_dim / 2;
        for (int b = 0; b < batch; b += 8) {
            auto *input_batch  = input_ptr + b * split_dim * count;
            auto *output_batch = output_ptr + b * output_dim * count;
            for (int c = 0; c < output_dim; c += 1) {
                auto *first_slice_ptr  = input_batch + c * count * 8;
                auto *second_slice_ptr = input_batch + (c + output_dim) * count * 8;
                auto *output_slice_ptr = output_batch + c * count * 8;
                for (int i = 0; i < count; ++i) {
                    Half8 a_v0 = Half8::load(first_slice_ptr + i * 8);
                    Half8 b_v0 = Half8::load(second_slice_ptr + i * 8);
                    b_v0       = Half8::neg(b_v0);
                    b_v0       = Half8::exp(b_v0);
                    b_v0       = b_v0 + one_v;
                    Half8 o_v0 = Half8::div(a_v0, b_v0);
                    Half8::save(output_slice_ptr + i * 8, o_v0);
                }
            }
        }
    }

    return TNN_OK;
}
#endif

}  // namespace TNN_NS
