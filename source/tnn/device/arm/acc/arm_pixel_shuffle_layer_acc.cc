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
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

DECLARE_ARM_ACC(PixelShuffle, LAYER_PIXEL_SHUFFLE);

Status ArmPixelShuffleLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto in_data_type   = inputs[0]->GetBlobDesc().data_type;
    if (in_data_type == DATA_TYPE_FLOAT) {
        return Exec<float>(inputs, outputs);
    } else if (in_data_type == DATA_TYPE_BFP16) {
        return Exec<bfp16_t>(inputs, outputs);
    } else {
        return Status(TNNERR_MODEL_ERR, "Error: layer acc dont support datatype");
    }
}

template<typename T>
static Status ExecFactor1(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input_dims  = inputs[0]->GetBlobDesc().dims;

    auto *input_ptr  = static_cast<T *>(inputs[0]->GetHandle().base);
    auto *output_ptr = static_cast<T *>(outputs[0]->GetHandle().base);

    int data_byte_size = DataTypeUtils::GetBytesSize(outputs[0]->GetBlobDesc().data_type);
    auto size_in_bytes = input_dims[0] * ROUND_UP(input_dims[1], 4) * input_dims[2] * input_dims[3] * data_byte_size;

    memcpy(output_ptr, input_ptr, size_in_bytes);

    return TNN_OK;
}

template<typename T>
static Status ExecFactor2(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs, void *workspace) {
    auto input_dims  = inputs[0]->GetBlobDesc().dims;
    auto output_dims = outputs[0]->GetBlobDesc().dims;

    auto ic    = input_dims[1];
    auto ic_r4 = ROUND_UP(input_dims[1], 4);
    auto ih    = input_dims[2];
    auto iw    = input_dims[3];
    auto oc    = output_dims[1];
    auto oc_r4 = ROUND_UP(output_dims[1], 4);
    auto oh    = output_dims[2];
    auto ow    = output_dims[3];

    auto input_plane     = ic * ih * iw;
    auto input_plane_r4  = ic_r4 * ih * iw;
    auto output_plane    = oc * oh * ow;
    auto output_plane_r4 = oc_r4 * oh * ow;

    auto *input_ptr  = static_cast<T *>(inputs[0]->GetHandle().base);
    auto *output_ptr = static_cast<T *>(outputs[0]->GetHandle().base);

    for (int b = 0; b < output_dims[0]; ++b) {
        auto workspace_data = reinterpret_cast<T *>(workspace) + b * output_plane;
        auto input_data     = input_ptr + b * input_plane_r4;

        for (int c = 0; c < oc; ++c) {
            auto workspace_data_c = workspace_data + c * oh * ow;
            auto input_data_c     = input_data + c * ih * iw * 4;
            for (int h = 0; h < oh; h += 2) {
                auto workspace_data_h = workspace_data_c + h * ow;
                auto input_data_h     = input_data_c + h / 2 * iw * 4;
                for (int w = 0; w < ow>>2<<2; w += 4) {
                    auto workspace_data_w0 = workspace_data_h + w;
                    auto workspace_data_w1 = workspace_data_w0 + ow;
                    auto input_data_w0     = input_data_h + w / 2 * 4;
                    auto input_data_w1     = input_data_w0 + 4;
                    Float4 c0 = Float4::load(input_data_w0);
                    Float4 c1 = Float4::load(input_data_w1);
                    Float2 temp0, temp1;
                    Float4::get_low(c0, temp0);
                    Float4::get_low(c1, temp1);
                    Float4 h0 = Float4::combine(temp0, temp1);
                    Float4::get_high(c0, temp0);
                    Float4::get_high(c1, temp1);
                    Float4 h1 = Float4::combine(temp0, temp1);
                    Float4::save(workspace_data_w0, h0);
                    Float4::save(workspace_data_w1, h1);
                }
                if (ow % 4 != 0) {
                    auto workspace_data_w0 = workspace_data_h + (ow>>2<<2);
                    auto workspace_data_w1 = workspace_data_w0 + ow;
                    auto input_data_w0     = input_data_h + (ow>>2<<2) / 2 * 4;
                    Float4 c0 = Float4::load(input_data_w0);
                    Float2 temp0, temp1;
                    Float4::get_low(c0, temp0);
                    Float4::get_high(c0, temp1);
                    workspace_data_w0[0] = temp0.value[0];
                    workspace_data_w0[1] = temp0.value[1];
                    workspace_data_w1[0] = temp1.value[0];
                    workspace_data_w1[1] = temp1.value[1];
                }
            }
        }

        auto output_data = output_ptr + b * output_plane_r4;
        PackC4(output_data, workspace_data, oh * ow, oc);
    }

    return TNN_OK;
}

template <typename T>
Status ArmPixelShuffleLayerAcc::Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param         = dynamic_cast<PixelShuffleLayerParam *>(param_);
    int upscale_factor = param->upscale_factor;

    if (upscale_factor == 1) {
        return ExecFactor1<T>(inputs, outputs);
    }

    int data_byte_size = DataTypeUtils::GetBytesSize(outputs[0]->GetBlobDesc().data_type);
    auto size_in_bytes = DimsVectorUtils::Count(outputs[0]->GetBlobDesc().dims) * data_byte_size;
    void *workspace    = context_->GetSharedWorkSpace(size_in_bytes);

    if (upscale_factor == 2) {
        return ExecFactor2<T>(inputs, outputs, workspace);
    } else {
        return Status(TNNERR_PARAM_ERR, "pixel shuffle upscale factor not support");
    }
}

REGISTER_ARM_ACC(PixelShuffle, LAYER_PIXEL_SHUFFLE);

}  // namespace TNN_NS
