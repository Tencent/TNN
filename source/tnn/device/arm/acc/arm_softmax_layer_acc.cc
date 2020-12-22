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
#include <cmath>
#include "tnn/device/arm/acc/arm_layer_acc.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/bfp16_utils.h"
#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {

DECLARE_ARM_ACC(Softmax, LAYER_SOFTMAX);

#define SoftmaxPreparation()                                   \
    auto in_data_type = inputs[0]->GetBlobDesc().data_type;    \
    auto axis = layer_param->axis;                             \
    auto input  = inputs[0];                                   \
    auto output = outputs[0];                                  \
    auto dims    = output->GetBlobDesc().dims;                 \
    auto width   = dims[3];                                    \
    auto height  = dims[2];                                    \
    auto batch   = dims[0];                                    \
    size_t count = width * height * batch * dims[1];           \
    int inside  = 1;                                           \
    int outside = 1;                                           \
    int channel = 1;                                           \
    for (int i = 1; i < axis; i++) {                           \
        outside *= dims[i];                                    \
    }                                                          \
    channel = dims[axis];                                      \
    for (int i = axis + 1; i < dims.size(); i++) {             \
        inside *= dims[i];                                     \
    }                                                          \
    auto step_y = channel * inside;

template <typename T>
Status ArmSoftmaxLayerAcc::Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    SoftmaxLayerParam *layer_param = dynamic_cast<SoftmaxLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    int data_byte_size = sizeof(float);
    SoftmaxPreparation();

    RawBuffer reorder_buffer(dims[1] * dims[2] * dims[3] * data_byte_size);
    RawBuffer max_value_buffer(inside * data_byte_size);
    RawBuffer sum_value_buffer(inside * data_byte_size);
    RawBuffer input_buffer(count * sizeof(float));
    RawBuffer output_buffer(count * sizeof(float));

    float *input_orign  = nullptr;
    float *output_orign = nullptr;
    if (in_data_type == DATA_TYPE_FLOAT) {
        input_orign  = reinterpret_cast<float *>(GetBlobHandlePtr(input->GetHandle()));
        output_orign = reinterpret_cast<float *>(GetBlobHandlePtr(output->GetHandle()));
    } else if (in_data_type == DATA_TYPE_BFP16) {
        bfp16_t *in_ptr = reinterpret_cast<bfp16_t *>(GetBlobHandlePtr(input->GetHandle()));
        input_orign     = input_buffer.force_to<float *>();
        output_orign    = output_buffer.force_to<float *>();
        ConvertFromBFP16ToFloat(in_ptr, input_orign, count);
    } else {
        return TNNERR_LAYER_ERR;
    }

    auto *max_value_ptr = max_value_buffer.force_to<float *>();
    auto *sum_value_ptr = sum_value_buffer.force_to<float *>();

    for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
        auto input_ptr  = input_orign + batch_idx * width * height * ROUND_UP(dims[1], 4);
        auto output_ptr = output_orign + batch_idx * width * height * ROUND_UP(dims[1], 4);

        UnpackC4(output_ptr, input_ptr, width * height, dims[1]);

        for (int y = 0; y < outside; y++) {
            auto src_y = output_ptr + y * step_y;
            auto dst_y = reorder_buffer.force_to<float *>() + y * step_y;
            memcpy(max_value_ptr, src_y, sizeof(float) * inside);

            auto src = src_y + inside;
            for (int c = 1; c < channel; ++c, src += inside) {
                for (int x = 0; x < inside; ++x) {
                    if (src[x] > max_value_ptr[x])
                        max_value_ptr[x] = src[x];
                }
            }

            memset(sum_value_ptr, 0, sizeof(float) * inside);
            src        = src_y;
            float *dst = dst_y;
            for (int c = 0; c < channel; ++c, src += inside, dst += inside) {
                for (int x = 0; x < inside; ++x) {
                    dst[x] = std::exp(src[x] - max_value_ptr[x]);
                }
            }

            dst = dst_y;
            for (int c = 0; c < channel; ++c, src += inside, dst += inside) {
                for (int x = 0; x < inside; ++x) {
                    sum_value_ptr[x] += dst[x];
                }
            }

            dst = dst_y;
            for (int c = 0; c < channel; ++c, dst += inside) {
                for (int x = 0; x < inside; ++x) {
                    dst[x] /= sum_value_ptr[x];
                }
            }
        }

        PackC4(output_ptr, reorder_buffer.force_to<float *>(), width * height, dims[1]);
    }

    if (in_data_type == DATA_TYPE_BFP16) {
        bfp16_t *out_ptr = reinterpret_cast<bfp16_t *>(GetBlobHandlePtr(output->GetHandle()));
        ConvertFromFloatToBFP16(output_orign, out_ptr, count);
    }

    return TNN_OK;
}

#if TNN_ARM82
template <>
Status ArmSoftmaxLayerAcc::Exec<fp16_t>(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    SoftmaxLayerParam *layer_param = dynamic_cast<SoftmaxLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    int data_byte_size = sizeof(fp16_t);
    SoftmaxPreparation();

    size_t reorder_size   = dims[1] * dims[2] * dims[3];
    size_t max_value_size = inside;
    size_t sum_value_size = inside;

    fp16_t *work_space = reinterpret_cast<fp16_t *>(
        context_->GetSharedWorkSpace((reorder_size + max_value_size + sum_value_size) * data_byte_size));

    fp16_t *reorder_buffer_ptr = work_space;
    fp16_t *max_value_ptr      = reorder_buffer_ptr + reorder_size;
    fp16_t *sum_value_ptr      = max_value_ptr + max_value_size;

    fp16_t *input_origin  = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(input->GetHandle()));
    fp16_t *output_origin = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(output->GetHandle()));

    for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
        auto input_ptr  = input_origin + batch_idx * width * height * ROUND_UP(dims[1], 8);
        auto output_ptr = output_origin + batch_idx * width * height * ROUND_UP(dims[1], 8);

        UnpackC8(output_ptr, input_ptr, width * height, dims[1]);

        for (int y = 0; y < outside; y++) {
            auto src_y = output_ptr + y * step_y;
            auto dst_y = reorder_buffer_ptr + y * step_y;
            memcpy(max_value_ptr, src_y, sizeof(fp16_t) * inside);

            auto src = src_y + inside;
            for (int c = 1; c < channel; ++c, src += inside) {
                int x = 0;
                for (; x <= inside - 8; x += 8) {
                    Half8 src_v = Half8::load(src + x);
                    Half8 max_v = Half8::load(max_value_ptr + x);
                    max_v = Half8::max(src_v, max_v);
                    Half8::save(max_value_ptr + x, max_v);
                }
                for (; x < inside; ++x) {
                    max_value_ptr[x] = src[x] > max_value_ptr[x] ? src[x] : max_value_ptr[x];
                }
            }

            memset(sum_value_ptr, 0, sizeof(fp16_t) * inside);
            src        = src_y;
            fp16_t *dst = dst_y;
            for (int c = 0; c < channel; ++c, src += inside, dst += inside) {
                int x = 0;
                for (; x <= inside - 8; x += 8) {
                    Half8 src_v = Half8::load(src + x);
                    Half8 max_v = Half8::load(max_value_ptr + x);
                    Half8 sum_v = Half8::load(sum_value_ptr + x);
                    Half8 dst_v = Half8::exp(src_v - max_v);
                    sum_v = sum_v + dst_v;
                    Half8::save(dst + x, dst_v);
                    Half8::save(sum_value_ptr + x, sum_v);
                }
                for (; x < inside; ++x) {
                    dst[x] = std::exp(src[x] - max_value_ptr[x]);
                    sum_value_ptr[x] += dst[x];
                }
            }

            for (int x = 0; x < inside; ++x) {
                sum_value_ptr[x] = 1 / sum_value_ptr[x];
            }

            dst = dst_y;
            for (int c = 0; c < channel; ++c, dst += inside) {
                int x = 0;
                for (; x < inside - 8; x += 8) {
                    Half8 dst_v = Half8::load(dst + x);
                    Half8 sum_v = Half8::load(sum_value_ptr + x);
                    dst_v = dst_v * sum_v;
                    Half8::save(dst + x, dst_v);
                }
                for (; x < inside; ++x) {
                    dst[x] *= sum_value_ptr[x];
                }
            }
        }

        PackC8(output_ptr, reorder_buffer_ptr, width * height, dims[1]);
    }
    return TNN_OK;
}
#endif

Status ArmSoftmaxLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto in_data_type = inputs[0]->GetBlobDesc().data_type;
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
        return Exec<fp16_t>(inputs, outputs);
    }
#endif
    else {
        return TNNERR_LAYER_ERR;
    }
}

REGISTER_ARM_ACC(Softmax, LAYER_SOFTMAX)
REGISTER_ARM_PRECISION_FP16(LAYER_SOFTMAX)

}  // namespace TNN_NS
