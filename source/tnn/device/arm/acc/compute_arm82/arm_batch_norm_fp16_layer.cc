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

#include "tnn/device/arm/acc/Half8.h"
#include "tnn/device/arm/acc/arm_batch_norm_layer_acc.h"
#include "tnn/utils/half_utils_inner.h"

namespace TNN_NS {

#if TNN_ARM82
Status ArmBatchNormLayerAcc::ExecFp16(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input       = inputs[0];
    auto output      = outputs[0];
    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;

    auto ic = dims_input[1], input_slice = UP_DIV(dims_input[1], 8);
    auto oc = dims_output[1], output_slice = UP_DIV(dims_output[1], 8);
    auto i_hw = DimsVectorUtils::Count(dims_input, 2);
    auto o_hw = DimsVectorUtils::Count(dims_output, 2);

    auto batch = dims_output[0];

    fp16_t *input_orign  = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(input->GetHandle()));
    fp16_t *output_orign = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(output->GetHandle()));

    fp16_t *k_data = buffer_scale_.force_to<fp16_t *>();
    fp16_t *b_data = buffer_bias_.force_to<fp16_t *>();

    auto src_z_step = i_hw * 8;
    auto dst_z_step = o_hw * 8;

    for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
        auto input_ptr  = input_orign + batch_idx * input_slice * 8 * i_hw;
        auto output_ptr = output_orign + batch_idx * output_slice * 8 * o_hw;

        if (!shared_channel_) {
            for (int dz = 0; dz < output_slice; dz++) {
                for (int x_i = 0; x_i < o_hw; x_i++) {
                    Half8 input_v  = Half8::load(input_ptr + dz * src_z_step + x_i * 8);
                    Half8 k_data_v = Half8::load(k_data + dz * 8);
                    Half8 b_data_v = Half8::load(b_data + dz * 8);
                    Half8::mla(b_data_v, input_v, k_data_v);
                    Half8::save(output_ptr + dz * dst_z_step + x_i * 8, b_data_v);
                }
            }
        } else {
            Half8 k_data_v = Half8(k_data[0]);
            Half8 b_data_v = Half8(b_data[0]);
            for (int dz = 0; dz < output_slice; dz++) {
                for (int x_i = 0; x_i < o_hw; x_i++) {
                    Half8 input_v = Half8::load(input_ptr + dz * src_z_step + x_i * 8);
                    Half8 dst_v   = b_data_v;
                    Half8::mla(dst_v, input_v, k_data_v);
                    Half8::save(output_ptr + dz * dst_z_step + x_i * 8, dst_v);
                }
            }
        }
    }

    return TNN_OK;
}
#endif

}  // namespace TNN_NS
