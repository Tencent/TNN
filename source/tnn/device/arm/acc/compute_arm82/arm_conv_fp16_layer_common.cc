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
#include "tnn/device/arm/acc/Half8.h"

#ifdef TNN_ARM82_A64 // aarch64 fp16
#define NEON_FP16CONV_TILE_HW (16)
#else // aarch32 fp16 or fp16 simu
#define NEON_FP16CONV_TILE_HW (8)
#endif

#ifdef TNN_ARM82_A64
static inline void _repack_half_16(fp16_t *dst_b, const fp16_t *src_b) {
    asm volatile (
        "ld4 {v0.8h, v1.8h, v2.8h, v3.8h}, [%0], #64\n\t"
        "ld4 {v4.8h, v5.8h, v6.8h, v7.8h}, [%0], #64\n\t"
        "ld4 {v8.8h, v9.8h, v10.8h, v11.8h}, [%0], #64\n\t"
        "ld4 {v12.8h, v13.8h, v14.8h, v15.8h}, [%0]\n\t"
        "uzp1 v16.8h, v0.8h, v4.8h\n\t"
        "uzp2 v20.8h, v0.8h, v4.8h\n\t"
        "uzp1 v17.8h, v1.8h, v5.8h\n\t"
        "uzp2 v21.8h, v1.8h, v5.8h\n\t"
        "uzp1 v18.8h, v2.8h, v6.8h\n\t"
        "uzp2 v22.8h, v2.8h, v6.8h\n\t"
        "uzp1 v19.8h, v3.8h, v7.8h\n\t"
        "uzp2 v23.8h, v3.8h, v7.8h\n\t"
        "uzp1 v0.8h,  v8.8h, v12.8h\n\t"
        "uzp2 v4.8h,  v8.8h, v12.8h\n\t"
        "uzp1 v1.8h,  v9.8h, v13.8h\n\t"
        "uzp2 v5.8h,  v9.8h, v13.8h\n\t"
        "uzp1 v2.8h,  v10.8h, v14.8h\n\t"
        "uzp2 v6.8h,  v10.8h, v14.8h\n\t"
        "uzp1 v3.8h,  v11.8h, v15.8h\n\t"
        "uzp2 v7.8h,  v11.8h, v15.8h\n\t"
        "str q16, [%2, #0]\n\t"
        "str q0,  [%2, #16]\n\t"
        "str q17, [%2, #32]\n\t"
        "str q1,  [%2, #48]\n\t"
        "str q18, [%2, #64]\n\t"
        "str q2,  [%2, #80]\n\t"
        "str q19, [%2, #96]\n\t"
        "str q3,  [%2, #112]\n\t"
        "str q20, [%2, #128]\n\t"
        "str q4,  [%2, #144]\n\t"
        "str q21, [%2, #160]\n\t"
        "str q5,  [%2, #176]\n\t"
        "str q22, [%2, #192]\n\t"
        "str q6,  [%2, #208]\n\t"
        "str q23, [%2, #224]\n\t"
        "str q7,  [%2, #240]\n\t"
        :"=r"(src_b)
        :"0"(src_b),"r"(dst_b)
        :"cc","memory","v0","v1","v2","v3","v4","v5","v6","v7","v8","v9",
        "v10","v11","v12","v13","v14","v15","v16","v17","v18","v19","v20",
        "v21","v22","v23"
    );
}
#endif

static inline void _repack_half_8(fp16_t *dst_b, const fp16_t *src_b) {
#ifdef TNN_ARM82_A64
    asm volatile (
        "ld4 {v0.8h, v1.8h, v2.8h, v3.8h}, [%0], #64\n\t"
        "ld4 {v4.8h, v5.8h, v6.8h, v7.8h}, [%0]\n\t"
        "uzp1 v8.8h,  v0.8h, v4.8h\n\t"
        "uzp2 v12.8h, v0.8h, v4.8h\n\t"
        "uzp1 v9.8h,  v1.8h, v5.8h\n\t"
        "uzp2 v13.8h, v1.8h, v5.8h\n\t"
        "uzp1 v10.8h, v2.8h, v6.8h\n\t"
        "uzp2 v14.8h, v2.8h, v6.8h\n\t"
        "uzp1 v11.8h, v3.8h, v7.8h\n\t"
        "uzp2 v15.8h, v3.8h, v7.8h\n\t"
        "str q8,  [%2, #0]\n\t"
        "str q9,  [%2, #16]\n\t"
        "str q10, [%2, #32]\n\t"
        "str q11, [%2, #48]\n\t"
        "str q12, [%2, #64]\n\t"
        "str q13, [%2, #80]\n\t"
        "str q14, [%2, #96]\n\t"
        "str q15, [%2, #112]\n\t"
        :"=r"(src_b)
        :"0"(src_b),"r"(dst_b)
        :"cc","memory","v0","v1","v2","v3","v4","v5","v6","v7","v8","v9",
        "v10","v11","v12","v13","v14","v15"
    );
#elif defined(TNN_ARM82_A32)
    asm volatile (
        "vld4.16 {d0, d2,  d4,  d6},  [%0]!\n\t"
        "vld4.16 {d1, d3,  d5,  d7},  [%0]!\n\t"
        "vld4.16 {d8, d10, d12, d14}, [%0]!\n\t"
        "vld4.16 {d9, d11, d13, d15}, [%0]\n\t"
        "vuzp.16 q0, q4\n\t"
        "vuzp.16 q1, q5\n\t"
        "vuzp.16 q2, q6\n\t"
        "vuzp.16 q3, q7\n\t"
        "vstr d0,  [%2, #0]\n\t"   "vstr d1,  [%2, #8]\n\t"
        "vstr d2,  [%2, #16]\n\t"  "vstr d3,  [%2, #24]\n\t"
        "vstr d4,  [%2, #32]\n\t"  "vstr d5,  [%2, #40]\n\t"
        "vstr d6,  [%2, #48]\n\t"  "vstr d7,  [%2, #56]\n\t"
        "vstr d8,  [%2, #64]\n\t"  "vstr d9,  [%2, #72]\n\t"
        "vstr d10, [%2, #80]\n\t"  "vstr d11, [%2, #88]\n\t"
        "vstr d12, [%2, #96]\n\t"  "vstr d13, [%2, #104]\n\t"
        "vstr d14, [%2, #112]\n\t" "vstr d15, [%2, #120]\n\t"
        :"=r"(src_b)
        :"0"(src_b),"r"(dst_b)
        :"cc","memory","q0","q1","q2","q3","q4","q5","q6","q7"
    );
#else
    fp16_t tmp[64];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            tmp[j * 8 + i] = src_b[i * 8 + j];
        }
    }
    for (int i = 0; i < 64; i++) {
        dst_b[i] = tmp[i];
    }
#endif
}

static inline void _repack_half_4(fp16_t *dst_b, const fp16_t *src_b) {
#ifdef TNN_ARM82_A64
    asm volatile (
        "ld4 {v0.4h, v1.4h, v2.4h, v3.4h}, [%0], #32\n\t"
        "ld4 {v4.4h, v5.4h, v6.4h, v7.4h}, [%0]\n\t"
        "uzp1 v8.4h,  v0.4h, v4.4h\n\t"
        "uzp2 v12.4h, v0.4h, v4.4h\n\t"
        "uzp1 v9.4h,  v1.4h, v5.4h\n\t"
        "uzp2 v13.4h, v1.4h, v5.4h\n\t"
        "uzp1 v10.4h, v2.4h, v6.4h\n\t"
        "uzp2 v14.4h, v2.4h, v6.4h\n\t"
        "uzp1 v11.4h, v3.4h, v7.4h\n\t"
        "uzp2 v15.4h, v3.4h, v7.4h\n\t"
        "str d8,  [%2, #0]\n\t"
        "str d9,  [%2, #8]\n\t"
        "str d10, [%2, #16]\n\t"
        "str d11, [%2, #24]\n\t"
        "str d12, [%2, #32]\n\t"
        "str d13, [%2, #40]\n\t"
        "str d14, [%2, #48]\n\t"
        "str d15, [%2, #56]\n\t"
        :"=r"(src_b)
        :"0"(src_b),"r"(dst_b)
        :"cc","memory","v0","v1","v2","v3","v4","v5","v6","v7","v8","v9",
        "v10","v11","v12","v13","v14","v15"
    );
#elif defined(TNN_ARM82_A32)
    asm volatile (
        "vld4.16 {d0, d1, d2, d3}, [%0]!\n\t"
        "vld4.16 {d4, d5, d6, d7}, [%0]\n\t"
        "vuzp.16 d0, d4\n\t"
        "vuzp.16 d1, d5\n\t"
        "vuzp.16 d2, d6\n\t"
        "vuzp.16 d3, d7\n\t"
        "vstr d0,  [%2, #0]\n\t"   "vstr d1,  [%2, #8]\n\t"
        "vstr d2,  [%2, #16]\n\t"  "vstr d3,  [%2, #24]\n\t"
        "vstr d4,  [%2, #32]\n\t"  "vstr d5,  [%2, #40]\n\t"
        "vstr d6,  [%2, #48]\n\t"  "vstr d7,  [%2, #56]\n\t"
        :"=r"(src_b)
        :"0"(src_b),"r"(dst_b)
        :"cc","memory","q0","q1","q2","q3"
    );
#else
    fp16_t tmp[32];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            tmp[j * 4 + i] = src_b[i * 8 + j];
        }
    }
    for (int i = 0; i < 32; i++) {
        dst_b[i] = tmp[i];
    }
#endif
}

static void load_repack_half_align(
    fp16_t *dst, 
    const fp16_t *src, 
    int dst_cnt, 
    int ic,
    int kernel_size) {
    int c = 0;
    for (; c <= ic - 8; c += 8) {
        for (int k = 0; k < kernel_size; k++) {
#ifdef TNN_ARM82_A64
            _repack_half_16(dst, src);
#else
            _repack_half_8(dst, src);
#endif
            src += 8 * NEON_FP16CONV_TILE_HW;
            dst += 8 * NEON_FP16CONV_TILE_HW;
        }
    }
    if (c < ic) {
        int c_eff = ic - c;
        for (int k = 0; k < kernel_size; k++) {
#ifdef TNN_ARM82_A64
            _repack_half_16(dst, src);
#else
            _repack_half_8(dst, src);
#endif
            src += 8 * NEON_FP16CONV_TILE_HW;
            dst += c_eff * NEON_FP16CONV_TILE_HW;
        }
    }
}

static void load_repack_half(
    fp16_t *dst, 
    const fp16_t *src, 
    int dst_cnt, 
    int ic,
    int kernel_size) {
    int dst_i = 0;
#ifdef TNN_ARM82_A64
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
#endif
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
    fp16_t *dst, 
    const fp16_t *src, 
    const ConvLayerParam *param, 
    size_t x_start, 
    size_t dst_cnt, 
    const ArmKernelParam *kparam) {

    auto src_s = src + x_start * 8;
    for (int c = 0; c <= kparam->ic_r8 - 8; c += 8) {
        auto src_c = src_s + c * kparam->ih * kparam->iw;
        auto dst_c = dst + c * NEON_FP16CONV_TILE_HW;
        memcpy(dst_c, src_c, dst_cnt * 8 * sizeof(fp16_t));
    }
}

/*
general img2col func
*/
static void img2col(
    fp16_t *dst, 
    const fp16_t *src, 
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
        const fp16_t *src;
    };

    tile_info tiles_info[NEON_FP16CONV_TILE_HW];
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
                        Half8::save(dst_c + i * 8, Half8::load(src_i));
                    }
                    dst_c += NEON_FP16CONV_TILE_HW * 8;
                }
            }
        }
    }
    // img2col memcpy normal mode
    else {
        // memset padding 0
        memset(dst, 0, kparam->ic_r8 * kh * kw * NEON_FP16CONV_TILE_HW * sizeof(fp16_t));
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
                        Half8::save(dst_fw, Half8::load(src_fw));
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
            Float2Half(filter_half.force_to<fp16_t *>(), conv_res->filter_handle.force_to<float *>(),
                       weight_nchw_count);
            // use int16_t to copy data, avoiding bad performance cased by fp16_t datatype in aarch32 fp16
            ConvertWeightsFromGOIHWToGOIHW64(filter_half.force_to<int16_t *>(), temp_buffer.force_to<int16_t *>(), group,
                                             ic, oc, conv_param->kernels[1], conv_param->kernels[0]);
        } else if (conv_res->filter_handle.GetDataType() == DATA_TYPE_HALF) {
            // soft fp16 -> fp32 -> hard fp16 TBD
            ConvertWeightsFromGOIHWToGOIHW64(conv_res->filter_handle.force_to<int16_t *>(),
                                             temp_buffer.force_to<int16_t *>(), group, ic, oc, conv_param->kernels[1],
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
                Float2Half(bias_nchw.force_to<fp16_t *>(), conv_res->bias_handle.force_to<float *>(), dims_output[1]);
                memcpy(temp_buffer.force_to<fp16_t *>(), bias_nchw.force_to<fp16_t *>(),
                       dims_output[1] * DataTypeUtils::GetBytesSize(DATA_TYPE_HALF));
            } else if (conv_res->bias_handle.GetDataType() == DATA_TYPE_HALF) {
                memcpy(temp_buffer.force_to<fp16_t *>(), conv_res->bias_handle.force_to<fp16_t *>(),
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

    // set tile blk size, which be limit to 16KB
    // 16 * 1024 / sizeof(fp16_t)
    int tile_blk = 8192 / (k_param_->ic_r8 * kernel_x * kernel_y);
    tile_blk = ROUND_UP(tile_blk, NEON_FP16CONV_TILE_HW);
    if (tile_blk < NEON_FP16CONV_TILE_HW) {
        tile_blk = NEON_FP16CONV_TILE_HW;
    }
    if (tile_blk > 512) {
        tile_blk = 512;
    }
    tile_blk_size = tile_blk;

    if (conv_param->activation_type == ActivationType_ReLU) {
        post_func_ = PostAddBiasRelu<fp16_t, fp16_t>;
    } else if (conv_param->activation_type == ActivationType_ReLU6) {
        post_func_ = PostAddBiasRelu6<fp16_t, fp16_t>;
    } else if (conv_param->activation_type == ActivationType_SIGMOID_MUL) {
        post_func_ = context_->GetPrecision() == PRECISION_NORMAL ? PostAddBiasSwish<fp16_t, fp16_t, false>
                                                                  : PostAddBiasSwish<fp16_t, fp16_t, true>;
    } else {
        post_func_ = PostAddBias<fp16_t, fp16_t>;
    }

    return TNN_OK;
}

template <>
void ArmConvFp16LayerCommon::PostExec<fp16_t>(const std::vector<Blob *> &outputs) {
    const int batch = outputs[0]->GetBlobDesc().dims[0];
    auto dst_origin = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(outputs[0]->GetHandle()));
    if (post_func_) {
        OMP_PARALLEL_FOR_
        for (int batch_idx = 0; batch_idx < batch; ++batch_idx) {
            auto output_ptr = dst_origin + batch_idx * k_param_->ow * k_param_->oh * k_param_->oc_r8;
            for (int dz = 0; dz < k_param_->oc_r8; dz += 8) {
                auto dst_z    = output_ptr + dz * k_param_->ow * k_param_->oh;
                fp16_t *bias_z = reinterpret_cast<fp16_t *>(k_param_->bias) + dz;
                post_func_(dst_z, bias_z, k_param_->ow * k_param_->oh, 1);
            }
        }
    }
}

void ArmConvFp16LayerCommon::PostExecNoBias(const std::vector<Blob *> &outputs) {
    const int batch = outputs[0]->GetBlobDesc().dims[0];
    auto dst_origin = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(outputs[0]->GetHandle()));
    if (post_func_) {
        OMP_PARALLEL_FOR_
        for (int batch_idx = 0; batch_idx < batch; ++batch_idx) {
            auto output_ptr = dst_origin + batch_idx * k_param_->ow * k_param_->oh * k_param_->oc_r8;
            for (int dz = 0; dz < k_param_->oc_r8; dz += 8) {
                auto dst_z    = output_ptr + dz * k_param_->ow * k_param_->oh;
                post_func_(dst_z, nullptr, k_param_->ow * k_param_->oh, 1);
            }
        }
    }
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

    fp16_t *input_data  = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(input->GetHandle()));
    fp16_t *output_data = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(output->GetHandle()));

    const int crs = ic * conv_param->kernels[1] * conv_param->kernels[0];
    const int crs_r8 = k_param_->ic_r8 * conv_param->kernels[1] * conv_param->kernels[0];
    const int tile_count = UP_DIV(k_param_->oh * k_param_->ow, tile_blk_size);

    int max_num_threads = OMP_MAX_THREADS_NUM_;
    size_t img2col_size = tile_blk_size * crs_r8;
    size_t repack_size = NEON_FP16CONV_TILE_HW * crs_r8;
    size_t workspace_size_per_thread = img2col_size + repack_size + NEON_KERNEL_EXTRA_LOAD;
    fp16_t *work_space = reinterpret_cast<fp16_t *>(
        context_->GetSharedWorkSpace(workspace_size_per_thread * max_num_threads * sizeof(fp16_t)));

    long act_type = conv_param->activation_type;
    if (conv_param->activation_type == ActivationType_SIGMOID_MUL) {
        act_type = 0;
    }

    for (int n = 0; n < batch; ++n) {
        const auto input_batch = input_data + n * k_param_->iw * k_param_->ih * k_param_->ic_r8;
        auto output_batch      = output_data + n * k_param_->ow * k_param_->oh * k_param_->oc_r8;

        OMP_PARALLEL_FOR_DYNAMIC_
        for (int t_idx = 0; t_idx < tile_count; t_idx++) {
            int thread_id          = OMP_TID_;
            auto workspace_per_thread = work_space + thread_id * workspace_size_per_thread;
            const int hw_start     = t_idx * tile_blk_size;
            const int real_hw_tile = MIN(k_param_->oh * k_param_->ow - hw_start, tile_blk_size);
            auto img2col_buffer = workspace_per_thread;
            auto output_kernel = output_batch + hw_start * 8;

            for (int i = 0; i < real_hw_tile; i += NEON_FP16CONV_TILE_HW) {
                int tile_eff = MIN(real_hw_tile - i, NEON_FP16CONV_TILE_HW);
                auto img2col_dst = img2col_buffer + i * crs_r8;
                img2col_func(img2col_dst, input_batch, conv_param, hw_start + i, tile_eff, k_param_.get());
            }

            auto repack_src = img2col_buffer;
            auto repack_dst = img2col_buffer;
            auto repack_tmp = img2col_buffer + crs_r8 * tile_blk_size + NEON_KERNEL_EXTRA_LOAD;

            int i = 0;
            for (; i <= real_hw_tile - NEON_FP16CONV_TILE_HW; i += NEON_FP16CONV_TILE_HW) {
                int tile_eff = MIN(real_hw_tile - i, NEON_FP16CONV_TILE_HW);
                // repack in-place if aligned
                load_repack_half_align(repack_dst + crs * i, repack_src + crs_r8 * i,
                                tile_eff, ic, conv_param->kernels[1] * conv_param->kernels[0]);
            }
            if (real_hw_tile > i) {
                int tile_eff = real_hw_tile - i;
                memcpy(repack_tmp, repack_src + crs_r8 * i, crs_r8 * NEON_FP16CONV_TILE_HW * sizeof(fp16_t));
                load_repack_half(repack_dst + crs * i, repack_tmp,
                                tile_eff, ic, conv_param->kernels[1] * conv_param->kernels[0]);
            }

            GEMM_FP16_N8(output_kernel, repack_dst, reinterpret_cast<fp16_t *>(k_param_->fil_ptr),
                        crs, 8 * k_param_->ow * k_param_->oh, k_param_->oc_r8, real_hw_tile, 
                        reinterpret_cast<fp16_t *>(k_param_->bias), act_type);
        }
    }

    if (conv_param->activation_type == ActivationType_SIGMOID_MUL) {
        PostExecNoBias(outputs);
    }

    return TNN_OK;
}
}  // namespace TNN_NS

#endif
