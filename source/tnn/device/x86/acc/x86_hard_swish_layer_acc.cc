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

#include "tnn/device/x86/acc/x86_layer_acc.h"
#include "tnn/device/x86/x86_device.h"
#include "immintrin.h"
#include <math.h>
#include <iostream>

namespace TNN_NS {

DECLARE_X86_ACC(HardSwish, X86_HARDSWISH_OP);

Status X86HardSwishLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status X86HardSwishLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    
    auto param = dynamic_cast<HardSwishLayerParam *>(param_);
    if (!param) {
        LOGE("Error: HardSwishLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: HardSwishLayerParam is nil");
    }

    Blob *input_blob0, *input_blob1;
    if (inputs.size() == 1) {
        input_blob0 = inputs[0];
        input_blob1 = inputs[0];
    } else {
        input_blob0 = inputs[0];
        input_blob1 = inputs[1];
    }

    const float alpha = param->alpha;
    const float beta  = param->beta;
    const float minV  = - beta / alpha;
    const float maxV  = (1.0f - beta) / alpha;

    auto input_dim0  = input_blob0->GetBlobDesc().dims;
    auto input_dim1  = input_blob1->GetBlobDesc().dims;

    const int batch = input_dim0[0];
    const int channel = input_dim0[1];
    const int channel_size = input_dim0[2] * input_dim0[3];

    auto input_ptr0 = reinterpret_cast<float *>(input_blob0->GetHandle().base);
    auto input_ptr1 = reinterpret_cast<float *>(input_blob1->GetHandle().base);

    auto output_ptr = reinterpret_cast<float *>(outputs[0]->GetHandle().base);

#ifdef __AVX2__
    register __m256 alpha_, beta_, zero_, one_, tmp00_, tmp01_, tmp02_, tmp03_, tmp10_, tmp11_, tmp12_, tmp13_;
    alpha_ = _mm256_set1_ps(alpha);
    beta_  = _mm256_set1_ps(beta);
    zero_  = _mm256_setzero_ps();
    one_   = _mm256_set1_ps(1.f);

    int total_size = batch * channel * channel_size;
    int tail = total_size - total_size % 8;
    int part_tail = tail / 4;
    const int offset1 = part_tail, offset2 = part_tail * 2, offset3 = part_tail * 3;

    if (input_dim0[2] == input_dim1[2]) {
        for (int index = 0; index < tail; index += 8) {
            tmp00_ = _mm256_loadu_ps(input_ptr0 + index);
            tmp10_ = _mm256_loadu_ps(input_ptr1 + index);
            tmp10_ = _mm256_fmadd_ps(tmp10_, alpha_, beta_);
            tmp10_ = _mm256_min_ps(tmp10_, one_);
            tmp10_ = _mm256_max_ps(tmp10_, zero_);
            tmp10_ = _mm256_mul_ps(tmp10_, tmp00_);
            _mm256_storeu_ps(output_ptr + index, tmp10_);
        }

        // build mask
        float mask[8] = {0.f};
        for (int i = 0; i < total_size % 8; i++) mask[i] = -1.f;
        __m256i mask_ = _mm256_loadu_si256((__m256i*)mask);

        tmp00_ = _mm256_maskload_ps(input_ptr0 + tail, mask_);
        tmp10_ = _mm256_maskload_ps(input_ptr1 + tail, mask_);

        tmp10_ = _mm256_fmadd_ps(tmp10_, alpha_, beta_);
        tmp10_ = _mm256_min_ps(tmp10_, one_);
        tmp10_ = _mm256_max_ps(tmp10_, zero_);
        tmp00_ = _mm256_mul_ps(tmp00_, tmp10_);
        _mm256_maskstore_ps(output_ptr + tail, mask_, tmp00_);
    } else {
        tail = channel_size - channel_size % 8;
        float mask[8] = {0.f};
        for (int i = 0; i < channel_size % 8; i++) mask[i] = -1.f;
        __m256i mask_ = _mm256_loadu_si256((__m256i*)mask);

        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < channel; c++) {
                float* input_data0 = input_ptr0 + (b * channel + c) * channel_size;
                float* input_data1 = input_ptr1 + (b * channel + c);
                float* output_data = output_ptr + (b * channel + c) * channel_size;
                float tmp = (*input_data1) * alpha + beta;
                tmp = std::max(std::min(tmp, 1.f), 0.f);
                tmp10_ = _mm256_set1_ps(tmp);
                for (int index = 0; index < tail; index += 8) {
                    tmp00_ = _mm256_loadu_ps(input_data0 + index);
                    tmp00_ = _mm256_mul_ps(tmp00_, tmp10_);
                    _mm256_storeu_ps(output_data + index, tmp00_);
                }
                tmp00_ = _mm256_maskload_ps(input_data0 + tail, mask_);
                tmp00_ = _mm256_mul_ps(tmp00_, tmp10_);
                _mm256_maskstore_ps(output_data + tail, mask_, tmp00_);
            }
        }
    }

#else
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channel; c++) {
            auto input_data0 = input_ptr0 + (b * channel + c) * channel_size;
            auto input_data1 = input_ptr1 + (b * channel + c) * channel_size;
            auto output_data = output_ptr + (b * channel + c) * channel_size;
            for (int index = 0; index < channel_size; index++) {
                float tmp = input_data1[index] * alpha + beta;
                tmp = std::min(tmp, 1.f);
                tmp = std::max(tmp, 0.f);
                output_data[index] = input_data0[index] * tmp;
            }
        }
    }
#endif
    return TNN_OK;
}

REGISTER_X86_ACC(HardSwish, LAYER_HARDSWISH);

}   // namespace TNN_NS
