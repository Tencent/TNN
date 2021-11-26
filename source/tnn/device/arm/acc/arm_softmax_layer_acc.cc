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

#include "tnn/device/arm/acc/arm_softmax_layer_acc.h"

#include <cmath>

#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/bfp16_utils.h"
#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {

static void SoftmaxChannelFunc(float *dst, float *src, int channel) {
    // max
    Float4 max_v = Float4(src[0]);
    float max    = src[0];
    int c        = 0;
    for (; c < channel - 4; c += 4) {
        max_v = Float4::max(Float4::load(src + c), max_v);
    }
    for (; c < channel; ++c) {
        max = std::max(max, src[c]);
    }
    for (int i = 0; i < 4; ++i) {
        max = std::max(max, max_v[i]);
    }
    // exp
    c = 0;
    for (; c < channel - 4; c += 4) {
        Float4::save(dst + c, Float4::exp(Float4::load(src + c) - Float4(max)));
    }
    for (; c < channel; ++c) {
        dst[c] = expf(src[c] - max);
    }
    // sum
    c            = 0;
    Float4 sum_v = Float4(0.0f);
    float sum    = 0.0f;
    for (; c < channel - 4; c += 4) {
        sum_v = Float4::load(dst + c) + sum_v;
    }
    for (; c < channel; ++c) {
        sum += dst[c];
    }
    for (int i = 0; i < 4; ++i) {
        sum += sum_v[i];
    }
    // div
    c                 = 0;
    float denominator = 1.0f / sum;
    for (; c < channel - 4; c += 4) {
        Float4::save(dst + c, Float4::load(dst + c) * denominator);
    }
    for (; c < channel; ++c) {
        dst[c] *= denominator;
    }
}

template <typename T>
Status ArmSoftmaxLayerAcc::Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    SoftmaxLayerParam *layer_param = dynamic_cast<SoftmaxLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    int data_byte_size = sizeof(float);
    SoftmaxPreparation();

    RawBuffer reorder_buffer;
    if (packed) {
        reorder_buffer = RawBuffer(dims[1] * hw * data_byte_size);
    }
    RawBuffer max_value_buffer(inside * data_byte_size);
    RawBuffer sum_value_buffer(inside * data_byte_size);
    RawBuffer input_buffer;
    RawBuffer output_buffer;

    float *input_orign  = nullptr;
    float *output_orign = nullptr;
    if (in_data_type == DATA_TYPE_FLOAT) {
        input_orign  = reinterpret_cast<float *>(GetBlobHandlePtr(input->GetHandle()));
        output_orign = reinterpret_cast<float *>(GetBlobHandlePtr(output->GetHandle()));
    } else if (in_data_type == DATA_TYPE_BFP16) {
        bfp16_t *in_ptr = reinterpret_cast<bfp16_t *>(GetBlobHandlePtr(input->GetHandle()));
        input_buffer    = RawBuffer(count * sizeof(float));
        output_buffer   = RawBuffer(count * sizeof(float));
        input_orign     = input_buffer.force_to<float *>();
        output_orign    = output_buffer.force_to<float *>();
        ConvertFromBFP16ToFloat(in_ptr, input_orign, count);
    } else {
        return TNNERR_LAYER_ERR;
    }

    auto *max_value_ptr = max_value_buffer.force_to<float *>();
    auto *sum_value_ptr = sum_value_buffer.force_to<float *>();

    for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
        auto input_ptr          = input_orign + batch_idx * hw * ROUND_UP(dims[1], packed ? 4 : 1);
        auto output_ptr         = output_orign + batch_idx * hw * ROUND_UP(dims[1], packed ? 4 : 1);
        auto reorder_buffer_ptr = output_ptr;

        if (packed) {
            UnpackC4(output_ptr, input_ptr, hw, dims[1]);
            input_ptr          = output_ptr;
            reorder_buffer_ptr = reorder_buffer.force_to<float *>();
        }
        if (inside == 1 && channel > 3) {
            for (int y = 0; y < outside; ++y) {
                auto src_y = input_ptr + y * step_y;
                auto dst_y = reorder_buffer_ptr + y * step_y;
                SoftmaxChannelFunc(dst_y, src_y, channel);
            }
        } else {
            for (int y = 0; y < outside; y++) {
                auto src_y = input_ptr + y * step_y;
                auto dst_y = reorder_buffer_ptr + y * step_y;
                memcpy(max_value_ptr, src_y, sizeof(float) * inside);
                // max
                auto src = src_y + inside;
                for (int c = 1; c < channel; ++c, src += inside) {
                    int x = 0;
                    for (; x < inside - 4; x += 4) {
                        Float4 src_v = Float4::load(src + x);
                        Float4 max_v = Float4::load(max_value_ptr + x);
                        max_v        = Float4::max(src_v, max_v);
                        Float4::save(max_value_ptr + x, max_v);
                    }
                    for (; x < inside; ++x) {
                        max_value_ptr[x] = src[x] > max_value_ptr[x] ? src[x] : max_value_ptr[x];
                    }
                }
                memset(sum_value_ptr, 0, sizeof(float) * inside);
                src        = src_y;
                float *dst = dst_y;
                for (int c = 0; c < channel; ++c, src += inside, dst += inside) {
                    int x = 0;
                    for (; x < inside - 4; x += 4) {
                        Float4 src_v = Float4::load(src + x);
                        Float4 max_v = Float4::load(max_value_ptr + x);
                        Float4 sum_v = Float4::load(sum_value_ptr + x);
                        Float4 dst_v = Float4::exp(src_v - max_v);
                        sum_v        = sum_v + dst_v;
                        Float4::save(dst + x, dst_v);
                        Float4::save(sum_value_ptr + x, sum_v);
                    }
                    for (; x < inside; ++x) {
                        dst[x] = expf(src[x] - max_value_ptr[x]);
                        sum_value_ptr[x] += dst[x];
                    }
                }
                dst = dst_y;
                for (int c = 0; c < channel; ++c, dst += inside) {
                    int x = 0;
                    for (; x < inside - 4; x += 4) {
                        Float4 dst_v = Float4::load(dst + x);
                        Float4 sum_v = Float4::load(sum_value_ptr + x);
                        dst_v        = Float4::div(dst_v, sum_v);
                        Float4::save(dst + x, dst_v);
                    }
                    for (; x < inside; ++x) {
                        dst[x] /= sum_value_ptr[x];
                    }
                }
            }
        }
        if (packed) {
            PackC4(output_ptr, reorder_buffer_ptr, hw, dims[1]);
        }
    }

    if (in_data_type == DATA_TYPE_BFP16) {
        bfp16_t *out_ptr = reinterpret_cast<bfp16_t *>(GetBlobHandlePtr(output->GetHandle()));
        ConvertFromFloatToBFP16(output_orign, out_ptr, count);
    }

    return TNN_OK;
}

Status ArmSoftmaxLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto in_data_type              = inputs[0]->GetBlobDesc().data_type;
    SoftmaxLayerParam *layer_param = dynamic_cast<SoftmaxLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    auto axis = layer_param->axis;
    if (axis == 0) {
        LOGE("ARM Softmax not support axis = 0\n");
        return Status(TNNERR_LAYER_ERR, "ARM Softmax not support axis = 0");
    }

    if (in_data_type == DATA_TYPE_FLOAT) {
        return Exec<float>(inputs, outputs);
    } else if (in_data_type == DATA_TYPE_BFP16) {
        return Exec<bfp16_t>(inputs, outputs);
    }
#if TNN_ARM82
    else if (in_data_type == DATA_TYPE_HALF) {
        return ExecFp16(inputs, outputs);
    }
#endif
    else {
        return TNNERR_LAYER_ERR;
    }
}

REGISTER_ARM_ACC(Softmax, LAYER_SOFTMAX)
REGISTER_ARM_PRECISION_FP16(LAYER_SOFTMAX)
REGISTER_ARM_LAYOUT(LAYER_SOFTMAX, DATA_FORMAT_NC4HW4)
REGISTER_ARM_LAYOUT(LAYER_SOFTMAX, DATA_FORMAT_NCHW)

}  // namespace TNN_NS
