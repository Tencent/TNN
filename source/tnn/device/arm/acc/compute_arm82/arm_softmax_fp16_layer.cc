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
#include "tnn/device/arm/acc/Half8.h"

namespace TNN_NS {

#if TNN_ARM82
Status ArmSoftmaxLayerAcc::ExecFp16(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
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
                    max_v       = Half8::max(src_v, max_v);
                    Half8::save(max_value_ptr + x, max_v);
                }
                for (; x < inside; ++x) {
                    max_value_ptr[x] = src[x] > max_value_ptr[x] ? src[x] : max_value_ptr[x];
                }
            }

            memset(sum_value_ptr, 0, sizeof(fp16_t) * inside);
            src         = src_y;
            fp16_t *dst = dst_y;
            for (int c = 0; c < channel; ++c, src += inside, dst += inside) {
                int x = 0;
                for (; x <= inside - 8; x += 8) {
                    Half8 src_v = Half8::load(src + x);
                    Half8 max_v = Half8::load(max_value_ptr + x);
                    Half8 sum_v = Half8::load(sum_value_ptr + x);
                    Half8 dst_v = Half8::exp(src_v - max_v);
                    sum_v       = sum_v + dst_v;
                    Half8::save(dst + x, dst_v);
                    Half8::save(sum_value_ptr + x, sum_v);
                }
                for (; x < inside; ++x) {
                    dst[x] = std::exp(src[x] - max_value_ptr[x]);
                    sum_value_ptr[x] += dst[x];
                }
            }

            for (int x = 0; x < inside; ++x) {
#ifdef TNN_ARM82_SIMU
                sum_value_ptr[x] = 1.0f / sum_value_ptr[x];
#else
                ((__fp16*)sum_value_ptr)[x] = 1.0f / ((__fp16*)sum_value_ptr)[x];
#endif
            }

            dst = dst_y;
            for (int c = 0; c < channel; ++c, dst += inside) {
                int x = 0;
                for (; x < inside - 8; x += 8) {
                    Half8 dst_v = Half8::load(dst + x);
                    Half8 sum_v = Half8::load(sum_value_ptr + x);
                    dst_v       = dst_v * sum_v;
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

}  // namespace TNN_NS
