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
#include "tnn/device/x86/x86_common.h"

#include <math.h>
#include <algorithm>

namespace TNN_NS {

DECLARE_X86_ACC(HardSwish, X86_HARDSWISH_OP);

Status X86HardSwishLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    
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

    auto input_dim0  = input_blob0->GetBlobDesc().dims;
    auto input_dim1  = input_blob1->GetBlobDesc().dims;

    int batch = input_dim0[0];
    int channel = 1;
    if (input_dim0.size() > 1) {
        channel = input_dim0[1];
    }
    int channel_size = 1;
    if (input_dim0.size() > 2) {
        channel_size = DimsVectorUtils::Count(input_dim0, 2);
    }

    auto input_ptr0 = handle_ptr<float *>(input_blob0->GetHandle());
    auto input_ptr1 = handle_ptr<float *>(input_blob1->GetHandle());

    auto output_ptr = handle_ptr<float *>(outputs[0]->GetHandle());

#ifdef __AVX__
    __m256 alpha_, beta_, zero_, one_, tmp00_, tmp01_, tmp02_, tmp03_, tmp10_, tmp11_, tmp12_, tmp13_;
    alpha_ = _mm256_set1_ps(alpha);
    beta_  = _mm256_set1_ps(beta);
    zero_  = _mm256_setzero_ps();
    one_   = _mm256_set1_ps(1.f);

    int total_size = batch * channel * channel_size;
    int tail = total_size - total_size % 32;
    int part_tail = tail / 4;
    const int offset1 = part_tail, offset2 = part_tail * 2, offset3 = part_tail * 3;
    
    if (input_dim0.size() > 2 && input_dim1.size() > 2) {
        if (input_dim0[2] == input_dim1[2]) {
            for (int index = 0; index < part_tail; index += 8) {
                tmp00_ = _mm256_loadu_ps(input_ptr0 + index);
                tmp01_ = _mm256_loadu_ps(input_ptr0 + index + offset1);
                tmp02_ = _mm256_loadu_ps(input_ptr0 + index + offset2);
                tmp03_ = _mm256_loadu_ps(input_ptr0 + index + offset3);
                tmp10_ = _mm256_loadu_ps(input_ptr1 + index);
                tmp11_ = _mm256_loadu_ps(input_ptr1 + index + offset1);
                tmp12_ = _mm256_loadu_ps(input_ptr1 + index + offset2);
                tmp13_ = _mm256_loadu_ps(input_ptr1 + index + offset3);

#ifdef __AVX2__
                tmp10_ = _mm256_fmadd_ps(tmp10_, alpha_, beta_);
                tmp11_ = _mm256_fmadd_ps(tmp11_, alpha_, beta_);
                tmp12_ = _mm256_fmadd_ps(tmp12_, alpha_, beta_);
                tmp13_ = _mm256_fmadd_ps(tmp13_, alpha_, beta_);
#else
                tmp10_ = _mm256_add_ps(_mm256_mul_ps(tmp10_, alpha_), beta_);
                tmp11_ = _mm256_add_ps(_mm256_mul_ps(tmp11_, alpha_), beta_);
                tmp12_ = _mm256_add_ps(_mm256_mul_ps(tmp12_, alpha_), beta_);
                tmp13_ = _mm256_add_ps(_mm256_mul_ps(tmp13_, alpha_), beta_);
#endif

                tmp10_ = _mm256_min_ps(tmp10_, one_);
                tmp11_ = _mm256_min_ps(tmp11_, one_);
                tmp12_ = _mm256_min_ps(tmp12_, one_);
                tmp13_ = _mm256_min_ps(tmp13_, one_);

                tmp10_ = _mm256_max_ps(tmp10_, zero_);
                tmp11_ = _mm256_max_ps(tmp11_, zero_);
                tmp12_ = _mm256_max_ps(tmp12_, zero_);
                tmp13_ = _mm256_max_ps(tmp13_, zero_);

                tmp10_ = _mm256_mul_ps(tmp10_, tmp00_);
                tmp11_ = _mm256_mul_ps(tmp11_, tmp01_);
                tmp12_ = _mm256_mul_ps(tmp12_, tmp02_);
                tmp13_ = _mm256_mul_ps(tmp13_, tmp03_);

                _mm256_storeu_ps(output_ptr + index,           tmp10_);
                _mm256_storeu_ps(output_ptr + index + offset1, tmp11_);
                _mm256_storeu_ps(output_ptr + index + offset2, tmp12_);
                _mm256_storeu_ps(output_ptr + index + offset3, tmp13_);
            }

            // build mask
            float mask[32] = {0.f};
            for (int i = 0; i < total_size % 32; i++) mask[i] = -1.f;
            __m256i mask0_ = _mm256_loadu_si256((__m256i*)mask);
            __m256i mask1_ = _mm256_loadu_si256((__m256i*)(mask + 8));
            __m256i mask2_ = _mm256_loadu_si256((__m256i*)(mask + 16));
            __m256i mask3_ = _mm256_loadu_si256((__m256i*)(mask + 24));

            tmp00_ = _mm256_maskload_ps(input_ptr0 + tail,      mask0_);
            tmp01_ = _mm256_maskload_ps(input_ptr0 + tail + 8,  mask1_);
            tmp02_ = _mm256_maskload_ps(input_ptr0 + tail + 16, mask2_);
            tmp03_ = _mm256_maskload_ps(input_ptr0 + tail + 24, mask3_);

            tmp10_ = _mm256_maskload_ps(input_ptr1 + tail,      mask0_);
            tmp11_ = _mm256_maskload_ps(input_ptr1 + tail + 8,  mask1_);
            tmp12_ = _mm256_maskload_ps(input_ptr1 + tail + 16, mask2_);
            tmp13_ = _mm256_maskload_ps(input_ptr1 + tail + 24, mask3_);

#ifdef __AVX2__
            tmp10_ = _mm256_fmadd_ps(tmp10_, alpha_, beta_);
            tmp11_ = _mm256_fmadd_ps(tmp11_, alpha_, beta_);
            tmp12_ = _mm256_fmadd_ps(tmp12_, alpha_, beta_);
            tmp13_ = _mm256_fmadd_ps(tmp13_, alpha_, beta_);
#else
            tmp10_ = _mm256_add_ps(_mm256_mul_ps(tmp10_, alpha_), beta_);
            tmp11_ = _mm256_add_ps(_mm256_mul_ps(tmp11_, alpha_), beta_);
            tmp12_ = _mm256_add_ps(_mm256_mul_ps(tmp12_, alpha_), beta_);
            tmp13_ = _mm256_add_ps(_mm256_mul_ps(tmp13_, alpha_), beta_);
#endif

            tmp10_ = _mm256_min_ps(tmp10_, one_);
            tmp11_ = _mm256_min_ps(tmp11_, one_);
            tmp12_ = _mm256_min_ps(tmp12_, one_);
            tmp13_ = _mm256_min_ps(tmp13_, one_);

            tmp10_ = _mm256_max_ps(tmp10_, zero_);
            tmp11_ = _mm256_max_ps(tmp11_, zero_);
            tmp12_ = _mm256_max_ps(tmp12_, zero_);
            tmp13_ = _mm256_max_ps(tmp13_, zero_);

            tmp10_ = _mm256_mul_ps(tmp00_, tmp10_);
            tmp11_ = _mm256_mul_ps(tmp01_, tmp11_);
            tmp12_ = _mm256_mul_ps(tmp02_, tmp12_);
            tmp13_ = _mm256_mul_ps(tmp03_, tmp13_);

            _mm256_maskstore_ps(output_ptr + tail,      mask0_, tmp10_);
            _mm256_maskstore_ps(output_ptr + tail + 8,  mask1_, tmp11_);
            _mm256_maskstore_ps(output_ptr + tail + 16, mask2_, tmp12_);
            _mm256_maskstore_ps(output_ptr + tail + 24, mask3_, tmp13_);

            return TNN_OK;
        }
    }

    tail = channel_size - channel_size % 8;
    float mask[8] = {0.f};
    for (int i = 0; i < channel_size % 8; i++) mask[i] = -1.f;
    __m256i mask_ = _mm256_loadu_si256((__m256i*)mask);

    if (channel % 4 == 0) {
        int c_offset = channel / 4;
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < channel / 4; c++) {
                float *single_data = input_ptr1 + b * channel + c;
                float tmp0 = single_data[0], tmp1 = single_data[c_offset], tmp2 = single_data[c_offset*2], tmp3 = single_data[c_offset*3];
                tmp0 = std::max(std::min(tmp0 * alpha + beta, 1.f), 0.f);
                tmp1 = std::max(std::min(tmp1 * alpha + beta, 1.f), 0.f);
                tmp2 = std::max(std::min(tmp2 * alpha + beta, 1.f), 0.f);
                tmp3 = std::max(std::min(tmp3 * alpha + beta, 1.f), 0.f);
                tmp10_ = _mm256_set1_ps(tmp0);
                tmp11_ = _mm256_set1_ps(tmp1);
                tmp12_ = _mm256_set1_ps(tmp2);
                tmp13_ = _mm256_set1_ps(tmp3);

                float* input_data00 = input_ptr0 + (b * channel + c) * channel_size;
                float* input_data01 = input_data00 + c_offset * channel_size;
                float* input_data02 = input_data01 + c_offset * channel_size;
                float* input_data03 = input_data02 + c_offset * channel_size;
                float* output_data0 = output_ptr + (b * channel + c) * channel_size;
                float* output_data1 = output_data0 + c_offset * channel_size;
                float* output_data2 = output_data1 + c_offset * channel_size;
                float* output_data3 = output_data2 + c_offset * channel_size;

                for (int index = 0; index < tail; index += 8) {
                    tmp00_ = _mm256_loadu_ps(input_data00 + index);
                    tmp01_ = _mm256_loadu_ps(input_data01 + index);
                    tmp02_ = _mm256_loadu_ps(input_data02 + index);
                    tmp03_ = _mm256_loadu_ps(input_data03 + index);

                    tmp00_ = _mm256_mul_ps(tmp00_, tmp10_);
                    tmp01_ = _mm256_mul_ps(tmp01_, tmp11_);
                    tmp02_ = _mm256_mul_ps(tmp02_, tmp12_);
                    tmp03_ = _mm256_mul_ps(tmp03_, tmp13_);

                    _mm256_storeu_ps(output_data0 + index, tmp00_);
                    _mm256_storeu_ps(output_data1 + index, tmp01_);
                    _mm256_storeu_ps(output_data2 + index, tmp02_);
                    _mm256_storeu_ps(output_data3 + index, tmp03_);
                }

                tmp00_ = _mm256_maskload_ps(input_data00 + tail, mask_);
                tmp01_ = _mm256_maskload_ps(input_data01 + tail, mask_);
                tmp02_ = _mm256_maskload_ps(input_data02 + tail, mask_);
                tmp03_ = _mm256_maskload_ps(input_data03 + tail, mask_);

                tmp00_ = _mm256_mul_ps(tmp00_, tmp10_);
                tmp01_ = _mm256_mul_ps(tmp01_, tmp11_);
                tmp02_ = _mm256_mul_ps(tmp02_, tmp12_);
                tmp03_ = _mm256_mul_ps(tmp03_, tmp13_);

                _mm256_maskstore_ps(output_data0 + tail, mask_, tmp00_);
                _mm256_maskstore_ps(output_data1 + tail, mask_, tmp01_);
                _mm256_maskstore_ps(output_data2 + tail, mask_, tmp02_);
                _mm256_maskstore_ps(output_data3 + tail, mask_, tmp03_);
            }
        }
    } else {
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
