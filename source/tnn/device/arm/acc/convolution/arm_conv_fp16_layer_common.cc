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
#if TNN_ARM82

#include "tnn/device/arm/acc/convolution/arm_conv_fp16_layer_common.h"

#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/omp_utils.h"

#define NEON_FP16CONV_TILE_HW (16)

static inline void _repack_half_16(__fp16 *dst_b, const __fp16 *src_b) {
    float16x8_t v[28];
    v[0] =  vld1q_f16(src_b + 0);
    v[1] =  vld1q_f16(src_b + 8);
    v[2] =  vld1q_f16(src_b + 16);
    v[3] =  vld1q_f16(src_b + 24);
    v[4] =  vld1q_f16(src_b + 32);
    v[5] =  vld1q_f16(src_b + 40);
    v[6] =  vld1q_f16(src_b + 48);
    v[7] =  vld1q_f16(src_b + 56);
    v[8] =  vld1q_f16(src_b + 64);
    v[9] =  vld1q_f16(src_b + 72);
    v[10] = vld1q_f16(src_b + 80);
    v[11] = vld1q_f16(src_b + 88);
    v[12] = vld1q_f16(src_b + 96);
    v[13] = vld1q_f16(src_b + 104);
    v[14] = vld1q_f16(src_b + 112);
    v[15] = vld1q_f16(src_b + 120);

    v[16] = vzip1q_f16(v[0],  v[4]);
    v[17] = vzip1q_f16(v[2],  v[6]);
    v[18] = vzip1q_f16(v[1],  v[5]);
    v[19] = vzip1q_f16(v[3],  v[7]);
    v[20] = vzip2q_f16(v[0],  v[4]);
    v[21] = vzip2q_f16(v[2],  v[6]);
    v[22] = vzip2q_f16(v[1],  v[5]);
    v[23] = vzip2q_f16(v[3],  v[7]);
    v[24] = vzip1q_f16(v[16], v[17]);
    v[25] = vzip1q_f16(v[18], v[19]);
    v[26] = vzip2q_f16(v[16], v[17]);
    v[27] = vzip2q_f16(v[18], v[19]);
    v[0]  = vzip1q_f16(v[24], v[25]);
    v[1]  = vzip2q_f16(v[24], v[25]);
    v[2]  = vzip1q_f16(v[26], v[27]);
    v[3]  = vzip2q_f16(v[26], v[27]);
    v[24] = vzip1q_f16(v[20], v[21]);
    v[25] = vzip1q_f16(v[22], v[23]);
    v[26] = vzip2q_f16(v[20], v[21]);
    v[27] = vzip2q_f16(v[22], v[23]);
    v[4]  = vzip1q_f16(v[24], v[25]);
    v[5]  = vzip2q_f16(v[24], v[25]);
    v[6]  = vzip1q_f16(v[26], v[27]);
    v[7]  = vzip2q_f16(v[26], v[27]);

    v[16] = vzip1q_f16(v[8],  v[12]);
    v[17] = vzip1q_f16(v[10], v[14]);
    v[18] = vzip1q_f16(v[9],  v[13]);
    v[19] = vzip1q_f16(v[11], v[15]);
    v[20] = vzip2q_f16(v[8],  v[12]);
    v[21] = vzip2q_f16(v[10], v[14]);
    v[22] = vzip2q_f16(v[9],  v[13]);
    v[23] = vzip2q_f16(v[11], v[15]);
    v[24] = vzip1q_f16(v[16], v[17]);
    v[25] = vzip1q_f16(v[18], v[19]);
    v[26] = vzip2q_f16(v[16], v[17]);
    v[27] = vzip2q_f16(v[18], v[19]);
    v[8]  = vzip1q_f16(v[24], v[25]);
    v[9]  = vzip2q_f16(v[24], v[25]);
    v[10] = vzip1q_f16(v[26], v[27]);
    v[11] = vzip2q_f16(v[26], v[27]);
    v[24] = vzip1q_f16(v[20], v[21]);
    v[25] = vzip1q_f16(v[22], v[23]);
    v[26] = vzip2q_f16(v[20], v[21]);
    v[27] = vzip2q_f16(v[22], v[23]);
    v[12] = vzip1q_f16(v[24], v[25]);
    v[13] = vzip2q_f16(v[24], v[25]);
    v[14] = vzip1q_f16(v[26], v[27]);
    v[15] = vzip2q_f16(v[26], v[27]);
    vst1q_f16(dst_b + 0,  v[0]);
    vst1q_f16(dst_b + 8,  v[8]);
    vst1q_f16(dst_b + 16, v[1]);
    vst1q_f16(dst_b + 24, v[9]);
    vst1q_f16(dst_b + 32, v[2]);
    vst1q_f16(dst_b + 40, v[10]);
    vst1q_f16(dst_b + 48, v[3]);
    vst1q_f16(dst_b + 56, v[11]);
    vst1q_f16(dst_b + 64, v[4]);
    vst1q_f16(dst_b + 72, v[12]);
    vst1q_f16(dst_b + 80, v[5]);
    vst1q_f16(dst_b + 88, v[13]);
    vst1q_f16(dst_b + 96, v[6]);
    vst1q_f16(dst_b + 104, v[14]);
    vst1q_f16(dst_b + 112, v[7]);
    vst1q_f16(dst_b + 120, v[15]);
}

static inline void _repack_half_8(__fp16 *dst_b, const __fp16 *src_b) {
    float16x8_t v[16];
    v[0] = vld1q_f16(src_b + 0);
    v[1] = vld1q_f16(src_b + 8);
    v[2] = vld1q_f16(src_b + 16);
    v[3] = vld1q_f16(src_b + 24);
    v[4] = vld1q_f16(src_b + 32);
    v[5] = vld1q_f16(src_b + 40);
    v[6] = vld1q_f16(src_b + 48);
    v[7] = vld1q_f16(src_b + 56);
    v[8]  = vzip1q_f16(v[0],  v[4]);
    v[9]  = vzip1q_f16(v[2],  v[6]);
    v[10] = vzip1q_f16(v[1],  v[5]);
    v[11] = vzip1q_f16(v[3],  v[7]);
    v[12] = vzip2q_f16(v[0],  v[4]);
    v[13] = vzip2q_f16(v[2],  v[6]);
    v[14] = vzip2q_f16(v[1],  v[5]);
    v[15] = vzip2q_f16(v[3],  v[7]);
    v[0]  = vzip1q_f16(v[8],  v[9]);
    v[1]  = vzip1q_f16(v[10], v[11]);
    v[2]  = vzip2q_f16(v[8],  v[9]);
    v[3]  = vzip2q_f16(v[10], v[11]);
    v[8]  = vzip1q_f16(v[0],  v[1]);
    v[9]  = vzip2q_f16(v[0],  v[1]);
    v[10] = vzip1q_f16(v[2],  v[3]);
    v[11] = vzip2q_f16(v[2],  v[3]);
    v[0]  = vzip1q_f16(v[12], v[13]);
    v[1]  = vzip1q_f16(v[14], v[15]);
    v[2]  = vzip2q_f16(v[12], v[13]);
    v[3]  = vzip2q_f16(v[14], v[15]);
    v[12] = vzip1q_f16(v[0],  v[1]);
    v[13] = vzip2q_f16(v[0],  v[1]);
    v[14] = vzip1q_f16(v[2],  v[3]);
    v[15] = vzip2q_f16(v[2],  v[3]);
    vst1q_f16(dst_b + 0,  v[8]);
    vst1q_f16(dst_b + 8,  v[9]);
    vst1q_f16(dst_b + 16, v[10]);
    vst1q_f16(dst_b + 24, v[11]);
    vst1q_f16(dst_b + 32, v[12]);
    vst1q_f16(dst_b + 40, v[13]);
    vst1q_f16(dst_b + 48, v[14]);
    vst1q_f16(dst_b + 56, v[15]);
}

static inline void _repack_half_4(__fp16 *dst_b, const __fp16 *src_b) {
    float16x4_t v[16];
    v[0] = vld1_f16(src_b + 0);
    v[1] = vld1_f16(src_b + 4);
    v[2] = vld1_f16(src_b + 8);
    v[3] = vld1_f16(src_b + 12);
    v[4] = vld1_f16(src_b + 16);
    v[5] = vld1_f16(src_b + 20);
    v[6] = vld1_f16(src_b + 24);
    v[7] = vld1_f16(src_b + 28);
    v[8]  = vzip1_f16(v[0],  v[4]);
    v[9]  = vzip1_f16(v[2],  v[6]);
    v[10] = vzip2_f16(v[0],  v[4]);
    v[11] = vzip2_f16(v[2],  v[6]);
    v[12] = vzip1_f16(v[1],  v[5]);
    v[13] = vzip1_f16(v[3],  v[7]);
    v[14] = vzip2_f16(v[1],  v[5]);
    v[15] = vzip2_f16(v[3],  v[7]);
    v[0]  = vzip1_f16(v[8],  v[9]);
    v[1]  = vzip2_f16(v[8],  v[9]);
    v[2]  = vzip1_f16(v[10], v[11]);
    v[3]  = vzip2_f16(v[10], v[11]);
    v[4]  = vzip1_f16(v[12], v[13]);
    v[5]  = vzip2_f16(v[12], v[13]);
    v[6]  = vzip1_f16(v[14], v[15]);
    v[7]  = vzip2_f16(v[14], v[15]);
    vst1_f16(dst_b + 0,  v[0]);
    vst1_f16(dst_b + 4,  v[1]);
    vst1_f16(dst_b + 8,  v[2]);
    vst1_f16(dst_b + 12, v[3]);
    vst1_f16(dst_b + 16, v[4]);
    vst1_f16(dst_b + 20, v[5]);
    vst1_f16(dst_b + 24, v[6]);
    vst1_f16(dst_b + 28, v[7]);
}

static void load_repack_half(
    __fp16 *dst, 
    const __fp16 *src, 
    int dst_cnt, 
    int ic,
    int kernel_size) {

    if (dst_cnt == NEON_FP16CONV_TILE_HW) {
        int c = 0;
        for (; c <= ic - 8; c += 8) {
            for (int k = 0; k < kernel_size; k++) {
                _repack_half_16(dst, src);
                src += 8 * NEON_FP16CONV_TILE_HW;
                dst += 8 * NEON_FP16CONV_TILE_HW;
            }
        }
        if (c < ic) {
            int c_eff = ic - c;
            for (int k = 0; k < kernel_size; k++) {
                _repack_half_16(dst, src);
                src += 8 * NEON_FP16CONV_TILE_HW;
                dst += c_eff * NEON_FP16CONV_TILE_HW;
            }
        }
        return;
    }

    int dst_i = 0;
    if (dst_cnt >= dst_i + 8) {
        auto src_p = src;
        int c = 0;
        for (; c <= ic - 8; c += 8) {
            for (int k = 0; k < kernel_size; k++) {
                _repack_half_8(dst, src_p);
                src_p += 8 * NEON_FP16CONV_TILE_HW;
                dst += 8 * 8;
            }
        }
        if (c < ic) {
            int c_eff = ic - c;
            for (int k = 0; k < kernel_size; k++) {
                _repack_half_8(dst, src_p);
                src_p += 8 * NEON_FP16CONV_TILE_HW;
                dst += c_eff * 8;
            }
        }
        src += 8 * 8;
        dst_i += 8;
    }
    if (dst_cnt >= dst_i + 4) {
        auto src_p = src;
        int c = 0;
        for (; c <= ic - 8; c += 8) {
            for (int k = 0; k < kernel_size; k++) {
                _repack_half_4(dst, src_p);
                src_p += 8 * NEON_FP16CONV_TILE_HW;
                dst += 8 * 4;
            }
        }
        if (c < ic) {
            int c_eff = ic - c;
            for (int k = 0; k < kernel_size; k++) {
                _repack_half_4(dst, src_p);
                src_p += 8 * NEON_FP16CONV_TILE_HW;
                dst += c_eff * 4;
            }
        }
        src += 4 * 8;
        dst_i += 4;
    }
    // when dst_cnt < 4, transpose tile = 4
    if (dst_cnt > dst_i) {
        auto src_p = src;
        int c = 0;
        for (; c <= ic - 8; c += 8) {
            for (int k = 0; k < kernel_size; k++) {
                _repack_half_4(dst, src_p);
                src_p += 8 * NEON_FP16CONV_TILE_HW;
                dst += 8 * 4;
            }
        }
        if (c < ic) {
            int c_eff = ic - c;
            for (int k = 0; k < kernel_size; k++) {
                _repack_half_4(dst, src_p);
                src_p += 8 * NEON_FP16CONV_TILE_HW;
                dst += c_eff * 4;
            }
        }
    }
}

namespace TNN_NS {
/*
ArmConvFp16LayerCommon as the last conv fp16 solution
*/
bool ArmConvFp16LayerCommon::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs) {
    if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_HALF) {
        return true;
    }
    return false;
}

ArmConvFp16LayerCommon::~ArmConvFp16LayerCommon() {}

/*
f1s1p0 img2col func
*/
static void img2col_f1s1p0(
    __fp16 *dst, 
    const __fp16 *src, 
    const ConvLayerParam *param, 
    size_t x_start, 
    size_t dst_cnt, 
    const ArmKernelParam *kparam) {

    auto src_s = src + x_start * 8;
    for (int c = 0; c <= kparam->ic_r8 - 8; c += 8) {
        auto src_c = src_s + c * kparam->ih * kparam->iw;
        auto dst_c = dst + c * NEON_FP16CONV_TILE_HW;
        memcpy(dst_c, src_c, dst_cnt * 8 * sizeof(__fp16));
    }
}

/*
general img2col func
*/
static void img2col(
    __fp16 *dst, 
    const __fp16 *src, 
    const ConvLayerParam *param, 
    size_t x_start, 
    size_t dst_cnt, 
    const ArmKernelParam *kparam) {

    int oh_start = ((int)x_start) / kparam->ow;
    int ow_start = ((int)x_start) % kparam->ow;
    int oh_end   = ((int)(x_start + dst_cnt - 1)) / kparam->ow;
    auto kh = param->kernels[1];
    auto kw = param->kernels[0];

    struct tile_info{
        int sfw;
        int efw;
        int sfh;
        int efh;
        const __fp16 *src;
    };

    tile_info tiles_info[16];
    tile_info *tiles_info_ptr = tiles_info;
    size_t dst_cnt_tmp = dst_cnt;
    int fast_mode_cnt = 0;
    // precompute src idx and ptr
    for (int oh = oh_start; oh <= oh_end; ++oh) {
        int sh = oh * param->strides[1] - param->pads[2];
        int sfh = MAX(0, (UP_DIV(-sh, param->dialations[1])));
        int efh = MIN(kh, UP_DIV(kparam->ih - sh, param->dialations[1]));
        int eff_ow = MIN(kparam->ow - ow_start, dst_cnt_tmp);
        auto src_sh = src + sh * kparam->iw * 8;
        int fast_flag = 1;
        if (efh - sfh != kh || oh_end != oh_start) {
            fast_flag = 0;
        }

        for (int i = 0; i < eff_ow; ++i) {
            int ow = ow_start + i;
            int sw = ow * param->strides[0] - param->pads[0];
            int sfw = MAX(0, (UP_DIV(-sw, param->dialations[0])));
            int efw = MIN(kw, UP_DIV(kparam->iw - sw, param->dialations[0]));
            if (efw - sfw != kw) {
                fast_flag = 0;
            }

            tiles_info_ptr[i].sfw = sfw;
            tiles_info_ptr[i].efw = efw;
            tiles_info_ptr[i].sfh = sfh;
            tiles_info_ptr[i].efh = efh;
            tiles_info_ptr[i].src = src_sh + sw * 8;
            if (fast_flag == 1) {
                fast_mode_cnt++;
            }
        }
        ow_start = 0;
        dst_cnt_tmp -= eff_ow;
        tiles_info_ptr += eff_ow;
    }

    size_t src_c_step = 8 * kparam->ih * kparam->iw;
    // img2col memcpy fast mode
    if (fast_mode_cnt == dst_cnt) {
        for (int c = 0; c <= kparam->ic_r8 - 8; c += 8) {
            auto src_c = tiles_info[0].src + c * kparam->ih * kparam->iw;
            auto dst_c = dst + c * NEON_FP16CONV_TILE_HW * kh * kw;
            for (int fh = 0; fh < kh; ++fh) {
                auto src_fh = src_c + fh * param->dialations[1] * kparam->iw * 8;
                for (int fw = 0; fw < kw; ++fw) {
                    auto src_fw = src_fh + fw * param->dialations[0] * 8;
                    for (int i = 0; i < dst_cnt; i++) {
                        auto src_i = src_fw + i * 8 * param->strides[0];
                        vst1q_f16(dst_c + i * 8, vld1q_f16(src_i));
                    }
                    dst_c += NEON_FP16CONV_TILE_HW * 8;
                }
            }
        }
    }
    // img2col memcpy normal mode
    else {
        for (int i = 0; i < dst_cnt; i++) {
            auto dst_i = dst + i * 8;
            for (int c = 0; c <= kparam->ic_r8 - 8; c += 8) {
                auto src_c = tiles_info[i].src + c * kparam->ih * kparam->iw;
                auto dst_c = dst_i + c * NEON_FP16CONV_TILE_HW * kh * kw;
                for (int fh = tiles_info[i].sfh; fh < tiles_info[i].efh; ++fh) {
                    auto src_fh = src_c + fh * param->dialations[1] * kparam->iw * 8;
                    auto dst_fh = dst_c + fh * NEON_FP16CONV_TILE_HW * 8 * kw;
                    for (int fw = tiles_info[i].sfw; fw < tiles_info[i].efw; ++fw) {
                        auto src_fw = src_fh + fw * param->dialations[0] * 8;
                        auto dst_fw = dst_fh + fw * NEON_FP16CONV_TILE_HW * 8;
                        vst1q_f16(dst_fw, vld1q_f16(src_fw));
                    }
                }
            }
        }
    }
}

Status ArmConvFp16LayerCommon::allocateBufferWeight(const std::vector<Blob *> &inputs,
                                                    const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);
    ConvLayerResource *conv_res = dynamic_cast<ConvLayerResource *>(resource_);
    CHECK_PARAM_NULL(conv_res);

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;

    if (!buffer_weight_.GetBytesSize()) {
        const int input_channel  = dims_input[1];
        const int output_channel = dims_output[1];

        int kw = conv_param->kernels[0];
        int kh = conv_param->kernels[1];

        // only support group == 1
        const int group = conv_param->group;
        if (group != 1) {
            LOGE("GROUP NOT SUPPORTED NOW\n");
            return Status(TNNERR_PARAM_ERR, "FP16 CONV COMMON GROUP > 1 NOT SUPPORT");
        }
        const int oc   = output_channel;
        const int ic   = input_channel;
        const int oc_8 = ROUND_UP(oc, 8);

        size_t weight_count   = group * oc_8 * ic * kh * kw;
        size_t data_byte_size = weight_count * DataTypeUtils::GetBytesSize(DATA_TYPE_HALF);
        RawBuffer temp_buffer(data_byte_size + NEON_KERNEL_EXTRA_LOAD);
        if (conv_res->filter_handle.GetDataType() == DATA_TYPE_FLOAT) {
            size_t weight_nchw_count = group * oc * ic * kh * kw;
            RawBuffer filter_half(weight_nchw_count * DataTypeUtils::GetBytesSize(DATA_TYPE_HALF));
            Float2Half(filter_half.force_to<__fp16 *>(), conv_res->filter_handle.force_to<float *>(),
                       weight_nchw_count);
            ConvertWeightsFromGOIHWToGOIHW64(filter_half.force_to<__fp16 *>(), temp_buffer.force_to<__fp16 *>(), group,
                                             ic, oc, conv_param->kernels[1], conv_param->kernels[0]);
        } else if (conv_res->filter_handle.GetDataType() == DATA_TYPE_HALF) {
            // soft fp16 -> fp32 -> hard fp16 TBD
            ConvertWeightsFromGOIHWToGOIHW64(conv_res->filter_handle.force_to<__fp16 *>(),
                                             temp_buffer.force_to<__fp16 *>(), group, ic, oc, conv_param->kernels[1],
                                             conv_param->kernels[0]);
        } else {
            LOGE("WEIGHT DATATYPE NOT SUPPORTED NOW\n");
            return Status(TNNERR_PARAM_ERR, "FP16 CONV COMMON ONLY SUPPORT WEIGHT DATATYPE FLOAT AND HALF");
        }

        buffer_weight_ = temp_buffer;
    }
    return TNN_OK;
}

Status ArmConvFp16LayerCommon::allocateBufferBias(const std::vector<Blob *> &inputs,
                                                  const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);
    ConvLayerResource *conv_res = dynamic_cast<ConvLayerResource *>(resource_);
    CHECK_PARAM_NULL(conv_res);

    auto dims_output = outputs[0]->GetBlobDesc().dims;
    if (!buffer_bias_.GetBytesSize()) {
        RawBuffer temp_buffer(ROUND_UP(dims_output[1], 8) * DataTypeUtils::GetBytesSize(DATA_TYPE_HALF));
        if (conv_param->bias) {
            if (conv_res->bias_handle.GetDataType() == DATA_TYPE_FLOAT) {
                RawBuffer bias_nchw(dims_output[1] * DataTypeUtils::GetBytesSize(DATA_TYPE_HALF));
                Float2Half(bias_nchw.force_to<__fp16 *>(), conv_res->bias_handle.force_to<float *>(), dims_output[1]);
                memcpy(temp_buffer.force_to<__fp16 *>(), bias_nchw.force_to<__fp16 *>(),
                       dims_output[1] * DataTypeUtils::GetBytesSize(DATA_TYPE_HALF));
            } else if (conv_res->bias_handle.GetDataType() == DATA_TYPE_HALF) {
                memcpy(temp_buffer.force_to<__fp16 *>(), conv_res->bias_handle.force_to<__fp16 *>(),
                       dims_output[1] * DataTypeUtils::GetBytesSize(DATA_TYPE_HALF));
            } else {
                LOGE("BIAS DATATYPE NOT SUPPORTED NOW\n");
                return Status(TNNERR_PARAM_ERR, "FP16 CONV COMMON ONLY SUPPORT BIAS DATATYPE FLOAT AND HALF");
            }
        }
        buffer_bias_ = temp_buffer;
    }

    return TNN_OK;
}

Status ArmConvFp16LayerCommon::Init(Context *context, LayerParam *param, LayerResource *resource,
                                    const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    RETURN_ON_NEQ(ArmLayerAcc::Init(context, param, resource, inputs, outputs), TNN_OK);
    RETURN_ON_NEQ(allocateBufferWeight(inputs, outputs), TNN_OK);
    RETURN_ON_NEQ(allocateBufferBias(inputs, outputs), TNN_OK);

    // init base k_param_
    k_param_->bias    = buffer_bias_.force_to<void *>();
    k_param_->fil_ptr = buffer_weight_.force_to<void *>();

    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);

    auto dims_input = inputs[0]->GetBlobDesc().dims;
    int kernel_x    = conv_param->kernels[0];
    int kernel_y    = conv_param->kernels[1];
    int stride_x    = conv_param->strides[0];
    int stride_y    = conv_param->strides[1];
    int pad_x       = conv_param->pads[0];
    int pad_y       = conv_param->pads[2];
    int dia_x       = conv_param->dialations[0];
    int dia_y       = conv_param->dialations[1];

    // im2col f1s1p0d1 fast mode
    bool f1s1p0d1 = kernel_x == 1 && kernel_y == 1 && stride_x == 1 && stride_y == 1 && pad_x == 0 && pad_y == 0 &&
                    dia_x == 1 && dia_y == 1;

    if (f1s1p0d1) {
        img2col_func = img2col_f1s1p0;
    } else {
        img2col_func = img2col;
    }

    return TNN_OK;
}

Status ArmConvFp16LayerCommon::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);
    auto input  = inputs[0];
    auto output = outputs[0];

    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;
    const int batch  = dims_output[0];
    auto ic          = dims_input[1];

    __fp16 *input_data  = reinterpret_cast<__fp16 *>(GetBlobHandlePtr(input->GetHandle()));
    __fp16 *output_data = reinterpret_cast<__fp16 *>(GetBlobHandlePtr(output->GetHandle()));

    const int crs = ic * conv_param->kernels[1] * conv_param->kernels[0];
    const int crs_r8 = k_param_->ic_r8 * conv_param->kernels[1] * conv_param->kernels[0];
    const int tile_count = UP_DIV(k_param_->oh * k_param_->ow, NEON_FP16CONV_TILE_HW);

    size_t img2col_size = NEON_FP16CONV_TILE_HW * crs_r8 * sizeof(__fp16);
    size_t transform_size = img2col_size;
    size_t out_tmp_size = k_param_->oc_r8 * NEON_FP16CONV_TILE_HW * sizeof(__fp16);
    __fp16 *work_space = reinterpret_cast<__fp16 *>(
        context_->GetSharedWorkSpace(img2col_size + transform_size + out_tmp_size + NEON_KERNEL_EXTRA_LOAD));

    for (int n = 0; n < batch; ++n) {
        const auto input_batch = input_data + n * k_param_->iw * k_param_->ih * k_param_->ic_r8;
        auto output_batch      = output_data + n * k_param_->ow * k_param_->oh * k_param_->oc_r8;

        // OMP_PARALLEL_FOR_GUIDED_
        for (int t_idx = 0; t_idx < tile_count; t_idx++) {
            // int thread_id          = OMP_TID_;
            const int hw_start     = t_idx * NEON_FP16CONV_TILE_HW;
            const int real_hw_tile = MIN(k_param_->oh * k_param_->ow - hw_start, NEON_FP16CONV_TILE_HW);
            auto img2col_buffer = work_space;
            auto output_kernel = output_batch + hw_start * 8;

            memset(img2col_buffer, 0, crs_r8 * NEON_FP16CONV_TILE_HW * sizeof(__fp16));
            img2col_func(img2col_buffer, input_batch, conv_param, hw_start, real_hw_tile, k_param_.get());

            auto repack_src = img2col_buffer;
            auto repack_dst = img2col_buffer + crs_r8 * NEON_FP16CONV_TILE_HW;
            auto outptr_tmp = img2col_buffer + crs_r8 * NEON_FP16CONV_TILE_HW * 2 + NEON_KERNEL_EXTRA_LOAD;
            // if aligned with TILE, do transpose inplace
            if (real_hw_tile == NEON_FP16CONV_TILE_HW) {
                repack_dst = img2col_buffer;
                outptr_tmp = img2col_buffer + crs_r8 * NEON_FP16CONV_TILE_HW + NEON_KERNEL_EXTRA_LOAD;
            }
            load_repack_half(repack_dst, repack_src, real_hw_tile, ic, conv_param->kernels[1] * conv_param->kernels[0]);

            GEMM_FP16_N8(outptr_tmp, repack_dst, reinterpret_cast<__fp16 *>(k_param_->fil_ptr),
                        crs, real_hw_tile * 8, k_param_->oc_r8, real_hw_tile, 
                        reinterpret_cast<__fp16 *>(k_param_->bias), conv_param->activation_type);

            for (int c = 0; c <= k_param_->oc_r8 - 8; c += 8) {
                auto src_b = outptr_tmp + c * real_hw_tile;
                auto dst_b = output_kernel + c * k_param_->ow * k_param_->oh;
                memcpy(dst_b, src_b, real_hw_tile * 8 * sizeof(__fp16));
            }
        }
    }
    return TNN_OK;
}

}  // namespace TNN_NS

#endif
