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

#include "tnn/device/arm/acc/arm_prelu_layer_acc.h"

#include <cmath>
#include "tnn/utils/half_utils_inner.h"
#include "tnn/device/arm/acc/Half8.h"

namespace TNN_NS {

#if TNN_ARM82
Status ArmPReluLayerAcc::ExecFp16(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<PReluLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    auto dims         = inputs[0]->GetBlobDesc().dims;
    const int channel = dims[1];
    const int hw      = DimsVectorUtils::Count(dims, 2);
    const int count   = dims[0] * ROUND_UP(dims[1], 8) * hw;

    const fp16_t *slope_data = buffer_slope_.force_to<fp16_t *>();

    fp16_t *input_data  = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(inputs[0]->GetHandle()));
    fp16_t *output_data = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(outputs[0]->GetHandle()));
    if (layer_param->channel_shared) {
        Half8 v_slope = Half8(slope_data[0]);
        Half8 v_zero  = Half8((fp16_t)(0.f));
        for (int n = 0; n < count; n += 8) {
            Half8 v_data = Half8::load(input_data + n);
            Half8 v_res  = Half8::bsl_clt(v_data, v_zero, v_data * v_slope, v_data);
            Half8::save(output_data + n, v_res);
        }
    } else {
        Half8 v_zero = Half8((fp16_t)(0.f));
        for (int batch_idx = 0; batch_idx < dims[0]; ++batch_idx) {
            auto input_ptr  = input_data + batch_idx * hw * ROUND_UP(channel, 8);
            auto output_ptr = output_data + batch_idx * hw * ROUND_UP(channel, 8);
            for (int dz = 0; dz < UP_DIV(channel, 8); ++dz) {
                auto *src_z         = input_ptr + dz * hw * 8;
                auto *dst_z         = output_ptr + dz * hw * 8;
                Half8 v_slope = Half8::load(slope_data + dz * 8);
                for (int p = 0; p < hw; p++) {
                    Half8 v_data = Half8::load(src_z + p * 8);
                    Half8 v_res  = Half8::bsl_clt(v_data, v_zero, v_data * v_slope, v_data);
                    Half8::save(dst_z + p * 8, v_res);
                }
            }
        }
    }

    return TNN_OK;
}
#endif

}  // namespace TNN_NS
