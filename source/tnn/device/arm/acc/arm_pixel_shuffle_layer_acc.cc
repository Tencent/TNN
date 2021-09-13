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
#include "tnn/utils/dims_utils.h"
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

template <typename T>
static Status ExecFactor1(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input_dims  = inputs[0]->GetBlobDesc().dims;

    auto *input_ptr  = reinterpret_cast<T *>(GetBlobHandlePtr(inputs[0]->GetHandle()));
    auto *output_ptr = reinterpret_cast<T *>(GetBlobHandlePtr(outputs[0]->GetHandle()));

    int data_byte_size = DataTypeUtils::GetBytesSize(outputs[0]->GetBlobDesc().data_type);
    auto size_in_bytes = input_dims[0] * ROUND_UP(input_dims[1], 4) * input_dims[2] * input_dims[3] * data_byte_size;

    memcpy(output_ptr, input_ptr, size_in_bytes);

    return TNN_OK;
}

#define PixelShufflePreparation                                                                                        \
    auto input_dims      = inputs[0]->GetBlobDesc().dims;                                                              \
    auto output_dims     = outputs[0]->GetBlobDesc().dims;                                                             \
    auto ic              = input_dims[1];                                                                              \
    auto ic_r4           = ROUND_UP(input_dims[1], 4);                                                                 \
    auto ih              = input_dims[2];                                                                              \
    auto iw              = input_dims[3];                                                                              \
    auto oc              = output_dims[1];                                                                             \
    auto oc_r4           = ROUND_UP(output_dims[1], 4);                                                                \
    auto oh              = output_dims[2];                                                                             \
    auto ow              = output_dims[3];                                                                             \
    auto input_plane     = ic * ih * iw;                                                                               \
    auto input_plane_r4  = ic_r4 * ih * iw;                                                                            \
    auto output_plane    = oc * oh * ow;                                                                               \
    auto output_plane_r4 = oc_r4 * oh * ow;                                                                            \
    auto *input_ptr      = reinterpret_cast<T *>(GetBlobHandlePtr(inputs[0]->GetHandle()));                            \
    auto *output_ptr     = reinterpret_cast<T *>(GetBlobHandlePtr(outputs[0]->GetHandle()));

template <typename T>
static void UnfoldPlane2x2(int oh, int ow, T *workspace_data_c, T *input_data_c) {
    for (int h = 0; h < oh; h += 2) {
        auto workspace_data_h = workspace_data_c + h * ow;
        auto input_data_h     = input_data_c + h * ow;
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
            workspace_data_w0[0] = temp0[0];
            workspace_data_w0[1] = temp0[1];
            workspace_data_w1[0] = temp1[0];
            workspace_data_w1[1] = temp1[1];
        }
    }
}

template <typename T>
static void ShuffleChannelLane(int upscale_factor, int oc, int ow, T *src_data_w, T *dst_data_w) {
    for (int rh = 0; rh < upscale_factor; ++rh) {
        for (int rw = 0; rw < upscale_factor; ++rw) {
            auto src_data_rw = src_data_w + rh * upscale_factor + rw;
            auto dst_data_rw = dst_data_w + rh * ow * oc + rw * oc;
            int stride    = oc;
            int stride_r4 = oc>>2<<2;
            int rc = 0;
            for (; rc < stride_r4; rc += 4) {
                Float4 src_value;
                auto src_val0 = *(src_data_rw + rc * upscale_factor * upscale_factor);
                auto src_val1 = *(src_data_rw + (rc + 1) * upscale_factor * upscale_factor);
                auto src_val2 = *(src_data_rw + (rc + 2) * upscale_factor * upscale_factor);
                auto src_val3 = *(src_data_rw + (rc + 3) * upscale_factor * upscale_factor);
                src_value.set_lane(src_val0, 0);
                src_value.set_lane(src_val1, 1);
                src_value.set_lane(src_val2, 2);
                src_value.set_lane(src_val3, 3);
                Float4::save(dst_data_rw + rc, src_value);
            }
            if (stride_r4 > 0 && (stride % 4 != 0)) {
                rc -= 4;
            }
            for (; rc < oc; ++rc) {
                auto src_data_c = src_data_rw + rc * upscale_factor * upscale_factor;
                auto dst_data_c = dst_data_rw + rc;
                *dst_data_c = *src_data_c;
            }
        }
    }
}

template <typename T>
static Status ExecFactor2(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs, void *workspace) {
    PixelShufflePreparation;

    for (int b = 0; b < output_dims[0]; ++b) {
        auto workspace_data = reinterpret_cast<T *>(workspace) + b * output_plane;
        auto input_data     = input_ptr + b * input_plane_r4;

        for (int c = 0; c < oc; ++c) {
            auto workspace_data_c = workspace_data + c * oh * ow;
            auto input_data_c     = input_data + c * ih * iw * 4;
            UnfoldPlane2x2(oh, ow, workspace_data_c, input_data_c);
        }

        auto output_data = output_ptr + b * output_plane_r4;
        PackC4(output_data, workspace_data, oh * ow, oc);
    }

    return TNN_OK;
}

template <typename T>
static Status ExecFactorCommon(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs, void *workspace,
                               int upscale_factor) {
    PixelShufflePreparation;

    for (int b = 0; b < output_dims[0]; ++b) {
        auto workspace_data_src = reinterpret_cast<T *>(workspace) + b * output_plane;
        auto workspace_data_dst = reinterpret_cast<T *>(workspace) + (output_dims[0] + b) * output_plane;
        auto input_data         = input_ptr + b * input_plane_r4;
        auto output_data        = output_ptr + b * output_plane_r4;

        UnpackC4ToNHWC(workspace_data_src, input_data, ih * iw, ic);

        for (int h = 0; h < ih; ++h) {
            auto dst_data_h = workspace_data_dst + h * iw * ic;
            auto src_data_h = workspace_data_src + h * iw * ic;
            for (int w = 0; w < iw; ++w) {
                auto dst_data_w = dst_data_h + w * ic / upscale_factor;
                auto src_data_w = src_data_h + w * ic;
                ShuffleChannelLane(upscale_factor, oc, ow, src_data_w, dst_data_w);
            }
        }

        PackC4FromNHWC(output_data, workspace_data_dst, oh * ow, oc);
    }

    return TNN_OK;
}

template <typename T>
Status ArmPixelShuffleLayerAcc::Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param         = dynamic_cast<PixelShuffleLayerParam *>(param_);
    CHECK_PARAM_NULL(param);
    int upscale_factor = param->upscale_factor;

    int data_byte_size = DataTypeUtils::GetBytesSize(outputs[0]->GetBlobDesc().data_type);
    auto size_in_bytes = DimsVectorUtils::Count(outputs[0]->GetBlobDesc().dims) * data_byte_size;

    if (upscale_factor == 1) {
        return ExecFactor1<T>(inputs, outputs);
    } else if (upscale_factor == 2) {
        void *workspace = context_->GetSharedWorkSpace(size_in_bytes);
        return ExecFactor2<T>(inputs, outputs, workspace);
    } else if (upscale_factor > 0) {
        void *workspace = context_->GetSharedWorkSpace(size_in_bytes * 2);
        return ExecFactorCommon<T>(inputs, outputs, workspace, upscale_factor);
    } else {
        return Status(TNNERR_PARAM_ERR, "pixel shuffle upscale factor not support");
    }
}

REGISTER_ARM_ACC(PixelShuffle, LAYER_PIXEL_SHUFFLE);
REGISTER_ARM_LAYOUT(LAYER_PIXEL_SHUFFLE, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
