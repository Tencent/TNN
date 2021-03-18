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

#include "tnn/device/arm/acc/arm_layer_acc.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_ARM_ACC(Shuffle, LAYER_SHUFFLE_CHANNEL);

static void inline ShuffleGeneral(float *output_ptr, const float *input_ptr, int group_row, int group_column,
                                  int channel, int area) {
    const int feature_map_size = channel * area;
    UnpackC4(output_ptr, input_ptr, area, channel);
    RawBuffer reorder_buffer(feature_map_size * sizeof(float));
    for (int i = 0; i < group_row; ++i)  // 2
    {
        for (int j = 0; j < group_column; ++j)  // 3
        {
            const float *p_i = output_ptr + (i * group_column + j) * area;
            float *p_o       = reorder_buffer.force_to<float *>() + (j * group_row + i) * area;
            memcpy(p_o, p_i, area * sizeof(float));
        }
    }
    PackC4(output_ptr, reorder_buffer.force_to<float *>(), area, channel);
}

#ifdef TNN_USE_NEON
static void inline Shuffle2(float *output_ptr, const float *input_ptr, int group_row, int group_column, int channel,
                            int area) {
    for (int c = 0; c < channel / 2; c += 4) {
        auto in_group_0  = input_ptr + c * area;
        auto in_group_1  = input_ptr + (c + channel / 2) * area;
        auto out_group_0 = output_ptr + c * 2 * area;
        auto out_group_1 = output_ptr + (c * 2 + 4) * area;
        for (int i = 0; i < area; i++) {
            float32x4_t v0 = vld1q_f32(in_group_0 + i * 4);
            float32x4_t v1 = vld1q_f32(in_group_1 + i * 4);

            float32x4x2_t vout = vzipq_f32(v0, v1);
            vst1q_f32(out_group_0 + i * 4, vout.val[0]);
            vst1q_f32(out_group_1 + i * 4, vout.val[1]);
        }
    }
}

#endif

Status ArmShuffleLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<ShuffleLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    auto input         = inputs[0];
    auto output        = outputs[0];
    auto dims          = input->GetBlobDesc().dims;
    DataType data_type = output->GetBlobDesc().data_type;
    const int area     = DimsVectorUtils::Count(dims, 2);

    int group_row    = param->group;
    int group_column = dims[1] / group_row;

    assert(dims[1] == (group_column * group_row));
    auto shuffle_func = ShuffleGeneral;
#ifdef TNN_USE_NEON
    if (param->group == 2 && dims[1] % 8 == 0) {
        shuffle_func = Shuffle2;
    }
#endif

    for (int n = 0; n < dims[0]; ++n) {
        if (data_type == DATA_TYPE_FLOAT) {
            auto input_ptr  = reinterpret_cast<float *>(GetBlobHandlePtr(input->GetHandle())) + n * area * ROUND_UP(dims[1], 4);
            auto output_ptr = reinterpret_cast<float *>(GetBlobHandlePtr(output->GetHandle())) + n * area * ROUND_UP(dims[1], 4);

            shuffle_func(output_ptr, input_ptr, group_row, group_column, dims[1], area);
        } else {
            return Status(TNNERR_LAYER_ERR, "NO IMPLEMENT FOR int8/bfp16 shuffle, in todo list");
        }
    }

    return TNN_OK;
}

REGISTER_ARM_ACC(Shuffle, LAYER_SHUFFLE_CHANNEL)
REGISTER_ARM_LAYOUT(LAYER_SHUFFLE_CHANNEL, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
