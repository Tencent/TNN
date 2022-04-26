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
#include "tnn/device/arm/acc/convolution/arm_conv_fp16_layer_3x3.h"

#ifdef TNN_USE_NEON
#include <arm_neon.h>
#endif
#include "tnn/device/arm/acc/compute/gemm_function.h"
#include "tnn/device/arm/acc/compute/winograd_function.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/omp_utils.h"
#include "tnn/device/arm/acc/Half8.h"

#ifdef TNN_ARM82_A64
#define NEON_GEMM_TILE_HW (16)
#else
#define NEON_GEMM_TILE_HW (8)
#endif

namespace TNN_NS {

#ifdef TNN_ARM82_A64
template <int stride>
static inline void _repack_half_16(__fp16 *dst_b, const __fp16 *src_b) {
    Half8 v[16];
    v[0] =  Half8::load(src_b + 0);
    v[1] =  Half8::load(src_b + stride);
    v[2] =  Half8::load(src_b + stride * 2);
    v[3] =  Half8::load(src_b + stride * 3);
    v[4] =  Half8::load(src_b + stride * 4);
    v[5] =  Half8::load(src_b + stride * 5);
    v[6] =  Half8::load(src_b + stride * 6);
    v[7] =  Half8::load(src_b + stride * 7);
    v[8] =  Half8::load(src_b + stride * 8);
    v[9] =  Half8::load(src_b + stride * 9);
    v[10] = Half8::load(src_b + stride * 10);
    v[11] = Half8::load(src_b + stride * 11);
    v[12] = Half8::load(src_b + stride * 12);
    v[13] = Half8::load(src_b + stride * 13);
    v[14] = Half8::load(src_b + stride * 14);
    v[15] = Half8::load(src_b + stride * 15);

    Half8::zip(v[0], v[4]);
    Half8::zip(v[2], v[6]);
    Half8::zip(v[1], v[5]);
    Half8::zip(v[3], v[7]);
    Half8::zip(v[0], v[2]);
    Half8::zip(v[1], v[3]);
    Half8::zip(v[4], v[6]);
    Half8::zip(v[5], v[7]);
    Half8::zip(v[0], v[1]);
    Half8::zip(v[2], v[3]);
    Half8::zip(v[4], v[5]);
    Half8::zip(v[6], v[7]);
    Half8::zip(v[8],  v[12]);
    Half8::zip(v[10], v[14]);
    Half8::zip(v[9],  v[13]);
    Half8::zip(v[11], v[15]);
    Half8::zip(v[8],  v[10]);
    Half8::zip(v[9],  v[11]);
    Half8::zip(v[12], v[14]);
    Half8::zip(v[13], v[15]);
    Half8::zip(v[8],  v[9]);
    Half8::zip(v[10], v[11]);
    Half8::zip(v[12], v[13]);
    Half8::zip(v[14], v[15]);

    Half8::save(dst_b + 0,  v[0]);
    Half8::save(dst_b + 8,  v[8]);
    Half8::save(dst_b + 16, v[1]);
    Half8::save(dst_b + 24, v[9]);
    Half8::save(dst_b + 32, v[2]);
    Half8::save(dst_b + 40, v[10]);
    Half8::save(dst_b + 48, v[3]);
    Half8::save(dst_b + 56, v[11]);
    Half8::save(dst_b + 64, v[4]);
    Half8::save(dst_b + 72, v[12]);
    Half8::save(dst_b + 80, v[5]);
    Half8::save(dst_b + 88, v[13]);
    Half8::save(dst_b + 96, v[6]);
    Half8::save(dst_b + 104, v[14]);
    Half8::save(dst_b + 112, v[7]);
    Half8::save(dst_b + 120, v[15]);
}
#endif

template <int stride>
static inline void _repack_half_8(fp16_t *dst_b, const fp16_t *src_b) {
    Half8 v[8];
    v[0] = Half8::load(src_b + 0);
    v[1] = Half8::load(src_b + stride);
    v[2] = Half8::load(src_b + stride * 2);
    v[3] = Half8::load(src_b + stride * 3);
    v[4] = Half8::load(src_b + stride * 4);
    v[5] = Half8::load(src_b + stride * 5);
    v[6] = Half8::load(src_b + stride * 6);
    v[7] = Half8::load(src_b + stride * 7);
    Half8::zip(v[0], v[4]);
    Half8::zip(v[2], v[6]);
    Half8::zip(v[1], v[5]);
    Half8::zip(v[3], v[7]);
    Half8::zip(v[0], v[2]);
    Half8::zip(v[1], v[3]);
    Half8::zip(v[4], v[6]);
    Half8::zip(v[5], v[7]);
    Half8::zip(v[0], v[1]);
    Half8::zip(v[2], v[3]);
    Half8::zip(v[4], v[5]);
    Half8::zip(v[6], v[7]);
    Half8::save(dst_b + 0,  v[0]);
    Half8::save(dst_b + 8,  v[1]);
    Half8::save(dst_b + 16, v[2]);
    Half8::save(dst_b + 24, v[3]);
    Half8::save(dst_b + 32, v[4]);
    Half8::save(dst_b + 40, v[5]);
    Half8::save(dst_b + 48, v[6]);
    Half8::save(dst_b + 56, v[7]);
}

template <int stride>
static inline void _repack_half_4(fp16_t *dst_b, const fp16_t *src_b) {
    Half4 v[8];
    v[0] = Half4::load(src_b + 0);
    v[1] = Half4::load(src_b + 4);
    v[2] = Half4::load(src_b + stride);
    v[3] = Half4::load(src_b + stride + 4);
    v[4] = Half4::load(src_b + stride * 2);
    v[5] = Half4::load(src_b + stride * 2 + 4);
    v[6] = Half4::load(src_b + stride * 3);
    v[7] = Half4::load(src_b + stride * 3 + 4);
    Half4::zip(v[0], v[4]);
    Half4::zip(v[2], v[6]);
    Half4::zip(v[1], v[5]);
    Half4::zip(v[3], v[7]);
    Half4::zip(v[0], v[2]);
    Half4::zip(v[4], v[6]);
    Half4::zip(v[1], v[3]);
    Half4::zip(v[5], v[7]);
    Half4::save(dst_b + 0,  v[0]);
    Half4::save(dst_b + 4,  v[2]);
    Half4::save(dst_b + 8,  v[4]);
    Half4::save(dst_b + 12, v[6]);
    Half4::save(dst_b + 16, v[1]);
    Half4::save(dst_b + 20, v[3]);
    Half4::save(dst_b + 24, v[5]);
    Half4::save(dst_b + 28, v[7]);
}

template <int src_unit_size>
static void load_repack_half(
    fp16_t *dst,
    const fp16_t *src,
    int dst_cnt,
    int z,
    int ic,
    int ic_r8) {
    if (dst_cnt == NEON_GEMM_TILE_HW) {
        auto repack_dst = dst + NEON_GEMM_TILE_HW * z;
        auto repack_src = src;
        for (int i = 0; i < src_unit_size; i++) {
            auto repack_dst_i = repack_dst + i * NEON_GEMM_TILE_HW * ic_r8;
            auto repack_src_i = repack_src + i * 8;
            if (src_unit_size == 16) {
#ifdef TNN_ARM82_A64
                _repack_half_16<128>(repack_dst_i, repack_src_i);
#else
                _repack_half_8<128>(repack_dst_i, repack_src_i);
#endif
            } else if (src_unit_size == 36) {
#ifdef TNN_ARM82_A64
                _repack_half_16<288>(repack_dst_i, repack_src_i);
#else
                _repack_half_8<288>(repack_dst_i, repack_src_i);
#endif
            }
        }
    } else {
        int x_i = 0;
#ifdef TNN_ARM82_A64
        if (x_i <= dst_cnt - 8) {
            auto repack_dst = dst + 8 * z;
            auto repack_src = src;
            for (int i = 0; i < src_unit_size; i++) {
                auto repack_dst_i = repack_dst + i * NEON_GEMM_TILE_HW * ic_r8;
                auto repack_src_i = repack_src + i * 8;
                if (src_unit_size == 16) {
                    _repack_half_8<128>(repack_dst_i, repack_src_i);
                } else if (src_unit_size == 36) {
                    _repack_half_8<288>(repack_dst_i, repack_src_i);
                }
            }
            x_i += 8;
        }
#endif
        if (x_i <= dst_cnt - 4) {
            auto repack_dst = dst + x_i * ic_r8 + 4 * z;
            auto repack_src = src + x_i * src_unit_size * 8;
            for (int i = 0; i < src_unit_size; i++) {
                auto repack_dst_i = repack_dst + i * NEON_GEMM_TILE_HW * ic_r8;
                auto repack_src_i = repack_src + i * 8;
                if (src_unit_size == 16) {
                    _repack_half_4<128>(repack_dst_i, repack_src_i);
                } else if (src_unit_size == 36) {
                    _repack_half_4<288>(repack_dst_i, repack_src_i);
                }
            }
            x_i += 4;
        }
        if (x_i < dst_cnt) {
            auto repack_dst = dst + x_i * ic_r8 + 4 * z;
            auto repack_src = src + x_i * src_unit_size * 8;
            for (int i = 0; i < src_unit_size; i++) {
                auto repack_dst_i = repack_dst + i * NEON_GEMM_TILE_HW * ic_r8;
                auto repack_src_i = repack_src + i * 8;
                if (src_unit_size == 16) {
                    _repack_half_4<128>(repack_dst_i, repack_src_i);
                } else if (src_unit_size == 36) {
                    _repack_half_4<288>(repack_dst_i, repack_src_i);
                }
            }
        }
    }
}

bool ArmConvFp16Layer3x3::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                 const std::vector<Blob *> &outputs) {
    if (!param) {
        return false;
    }

    if (inputs[0]->GetBlobDesc().data_type != DATA_TYPE_HALF) {
        return false;
    }

    if (param->group != 1 || param->dialations[0] != 1 || param->dialations[1] != 1 || param->strides[0] != 1 ||
        param->kernels[0] != param->kernels[1] || param->strides[1] != 1 ||
        (inputs[0]->GetBlobDesc().dims[1] < 8 && outputs[0]->GetBlobDesc().dims[1] < 8)) {
        return false;
    }

    if (!SelectWinograd(param, inputs, outputs)) {
        return false;
    }

    return true;
}

int ArmConvFp16Layer3x3::SelectWinograd(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                    const std::vector<Blob *> &outputs) {
    if (!param) {
        return 0;
    }

    // gemm: only use gemm
    // winograd_unit2: use winograd f(2,3)
    // winograd_unit4 or others:
    // use winograd f(4,3) or f(2,3)
    if (param->extra_config.count("arm_fp16_gemm")) {
        return 0;
    }

    int ic          = inputs[0]->GetBlobDesc().dims[1];
    int oc          = outputs[0]->GetBlobDesc().dims[1];
    int kernel_size = param->kernels[0];
    int ow          = outputs[0]->GetBlobDesc().dims[3];
    int oh          = outputs[0]->GetBlobDesc().dims[2];

    if (kernel_size != 3) {
        return 0;
    }

    int dst_unit      = 2;
    float max_rate    = 1.f;
    float origin_cost = (float)ow * oh * (float)ROUND_UP(ic, 8) * ROUND_UP(oc, 8) * kernel_size * kernel_size;

    // only support F(2x2, 3x3) and F(4x4, 3x3)
    for (int u = 2; u <= 4; u += 2) {
        float src_unit = (float)(u + kernel_size - 1);

        // winograd cost = src transform + gemm + dst transform
        float winograd_cost =
            (2 * src_unit * src_unit * src_unit * ROUND_UP(ic, 8) +
             src_unit * src_unit * ROUND_UP(ic, 8) * ROUND_UP(oc, 8) + 2 * src_unit * u * u * ROUND_UP(oc, 8)) *
            (UP_DIV(ow, u) * UP_DIV(oh, u));

        float acc_rate = origin_cost / winograd_cost;

        if (acc_rate > max_rate * 1.1f) {
            max_rate = acc_rate;
            dst_unit = u;
        }
    }

    // 10% penalty, winograd will result in more cache miss
    if (max_rate < 1.1f) {
        return 0;
    }

    if (param->extra_config.count("arm_fp16_winograd_unit2")) {
        dst_unit = 2;
    }

    return dst_unit;
}

ArmConvFp16Layer3x3::~ArmConvFp16Layer3x3() {}

Status ArmConvFp16Layer3x3::allocateBufferWeight(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);
    ConvLayerResource *conv_res = dynamic_cast<ConvLayerResource *>(resource_);
    CHECK_PARAM_NULL(conv_res);

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;

    if (!buffer_weight_.GetBytesSize()) {
        const int ic  = dims_input[1];
        const int oc = dims_output[1];

        const int kw = conv_param->kernels[0];
        const int kh = conv_param->kernels[1];

        const float *src = conv_res->filter_handle.force_to<float *>();
        size_t data_byte_size = DataTypeUtils::GetBytesSize(DATA_TYPE_HALF);

        dst_unit_ = SelectWinograd(conv_param, inputs, outputs);
        src_unit_ = dst_unit_ + kw - 1;

        const size_t weight_count = src_unit_ * src_unit_ * k_param_->oc_r8 * k_param_->ic_r8;
        const size_t weight_nchw_count = oc * ic * kh * kw;

        RawBuffer pack_weight_f32(weight_count * DataTypeUtils::GetBytesSize(DATA_TYPE_FLOAT));
        RawBuffer pack_weight_f16(weight_count * data_byte_size + NEON_KERNEL_EXTRA_LOAD);
        // use float to transform weight, may be helpful to higher precision
        // in aarch32 fp16, need to avoid using half_t to compute weight, because it is slow
        switch (dst_unit_) {
            case 2:
                WeightTransformHalf4x4(src, pack_weight_f32.force_to<float *>(), 3, ic, oc);
                break;
            case 4:
                WeightTransformHalf6x6(src, pack_weight_f32.force_to<float *>(), 3, ic, oc);
                break;
            default:
                LOGE("Unsupport winograd dst unit\n");
                break;
        }
        Float2Half(pack_weight_f16.force_to<fp16_t *>(), pack_weight_f32.force_to<float *>(), weight_count);
        buffer_weight_ = pack_weight_f16;
    }

    return TNN_OK;
}

Status ArmConvFp16Layer3x3::Init(Context *context, LayerParam *param, LayerResource *resource,
                             const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    RETURN_ON_NEQ(ArmConvFp16LayerCommon::Init(context, param, resource, inputs, outputs), TNN_OK);
    auto in_data_type = inputs[0]->GetBlobDesc().data_type;

    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    if (conv_param) {
        if (in_data_type == DATA_TYPE_HALF) {
            if (dst_unit_ == 2) {
                SrcTransformFunc_ = SrcTransformInOne4x4Fp16;
                DstTransformFunc_ = DstTransformInOne4x2Fp16;
            } else if (dst_unit_ == 4) {
                SrcTransformFunc_ = SrcTransformInOne6x6Fp16;
                DstTransformFunc_ = DstTransformInOne6x4Fp16;
            } else {
                return TNNERR_LAYER_ERR;
            }
        } else {
            return TNNERR_LAYER_ERR;
        }
    }
    return TNN_OK;
}

Status ArmConvFp16Layer3x3::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (dst_unit_ == 2) {
        return Exec<2, 4>(inputs, outputs);
    } else if (dst_unit_ == 4) {
        return Exec<4, 6>(inputs, outputs);
    } else {
        return TNNERR_LAYER_ERR;
    }
}

template <int dst_unit, int src_unit>
Status ArmConvFp16Layer3x3::Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    auto input                 = inputs[0];
    auto output                = outputs[0];

    size_t data_byte_size = DataTypeUtils::GetBytesSize(DATA_TYPE_HALF);

    const int batch = output->GetBlobDesc().dims[0];
    int ic = input->GetBlobDesc().dims[1];

    auto w_unit      = UP_DIV(k_param_->ow, dst_unit);
    auto h_unit      = UP_DIV(k_param_->oh, dst_unit);
    auto tile_count  = UP_DIV(w_unit * h_unit, NEON_GEMM_TILE_HW);

    const fp16_t *src_origin = reinterpret_cast<const fp16_t *>(GetBlobHandlePtr(input->GetHandle()));
    fp16_t *dst_origin = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(output->GetHandle()));

    int max_num_threads           = OMP_MAX_THREADS_NUM_;
    size_t fake_bias_size         = k_param_->oc_r8 * data_byte_size;

    size_t src_pad_buf_per_thread = src_unit * src_unit * 8;
    size_t src_pad_buf_size       = src_pad_buf_per_thread * max_num_threads * data_byte_size;

    size_t src_trans_tmp_per_thread = 8 * src_unit * src_unit * NEON_GEMM_TILE_HW;
    size_t src_trans_size           = k_param_->ic_r8 * src_unit * src_unit * NEON_GEMM_TILE_HW;
    size_t dst_trans_size           = k_param_->oc_r8 * src_unit * src_unit * NEON_GEMM_TILE_HW;
    size_t work_buf_size            = (src_trans_tmp_per_thread * max_num_threads + src_trans_size + dst_trans_size) * data_byte_size;

    fp16_t *work_space = reinterpret_cast<fp16_t *>(
        context_->GetSharedWorkSpace(fake_bias_size + src_pad_buf_size + work_buf_size + NEON_KERNEL_EXTRA_LOAD));

    fp16_t *fake_bias      = work_space;
    fp16_t *src_pad_buffer = work_space + fake_bias_size / data_byte_size;
    fp16_t *work_buf       = src_pad_buffer + src_pad_buf_size / data_byte_size;

    // memset fake bias data to get correct results
    memset(fake_bias, 0, fake_bias_size);

    if (DstTransformFunc_ == nullptr || SrcTransformFunc_ == nullptr) {
        return TNNERR_COMMON_ERROR;
    }

    struct tile_info {
        int src_sy;
        int src_ey;
        int src_sx;
        int src_ex;
        int src_loc;

        int dst_ey;
        int dst_ex;
        int dst_loc;
    };
    tile_info tiles_info[NEON_GEMM_TILE_HW];

    for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
        auto input_ptr  = src_origin + batch_idx * k_param_->iw * k_param_->ih * k_param_->ic_r8;
        auto output_ptr = dst_origin + batch_idx * k_param_->ow * k_param_->oh * k_param_->oc_r8;

        for (int t_idx = 0; t_idx < tile_count; t_idx++) {
            auto src_trans_buf = work_buf;
            auto repack_buf    = src_trans_buf + src_trans_tmp_per_thread * max_num_threads;
            auto dst_trans_buf = repack_buf + src_trans_size + NEON_KERNEL_EXTRA_LOAD / data_byte_size;

            int x_idx    = t_idx * NEON_GEMM_TILE_HW;
            int x_remain = w_unit * h_unit - x_idx;
            int x_c      = x_remain > NEON_GEMM_TILE_HW ? NEON_GEMM_TILE_HW : x_remain;

            size_t src_z_step = k_param_->iw * k_param_->ih;
            size_t dst_z_step = x_c * src_unit * src_unit;

            // pre-compute index and offset
            for (int x_i = 0; x_i < x_c; x_i++) {
                int idx = x_idx + x_i;
                int w_idx = idx % w_unit;
                int h_idx = idx / w_unit;

                int src_x = w_idx * dst_unit - conv_param->pads[0];
                int src_y = h_idx * dst_unit - conv_param->pads[2];
                int dst_x = w_idx * dst_unit;
                int dst_y = h_idx * dst_unit;

                tiles_info[x_i].src_sy  = MAX(0, src_y) - src_y;
                tiles_info[x_i].src_ey  = MIN(src_y + src_unit, k_param_->ih) - src_y;
                tiles_info[x_i].src_sx  = MAX(0, src_x) - src_x;
                tiles_info[x_i].src_ex  = MIN(src_x + src_unit, k_param_->iw) - src_x;
                tiles_info[x_i].src_loc = (src_x + src_y * k_param_->iw) * 8;

                tiles_info[x_i].dst_ey  = MIN(dst_y + dst_unit, k_param_->oh) - dst_y;
                tiles_info[x_i].dst_ex  = MIN(dst_x + dst_unit, k_param_->ow) - dst_x;
                tiles_info[x_i].dst_loc = (dst_x + dst_y * k_param_->ow) * 8;
            }

            OMP_PARALLEL_FOR_
            for (int z = 0; z <= k_param_->ic_r8 - 8; z += 8) {
                int tid         = OMP_TID_;
                auto mid_buffer = src_pad_buffer + tid * src_pad_buf_per_thread;
                auto src_z      = input_ptr + z * src_z_step;
                auto dst_z      = src_trans_buf + tid * src_trans_tmp_per_thread;
                for (int x_i = 0; x_i < x_c; x_i++) {
                    int sy    = tiles_info[x_i].src_sy;
                    int ey    = tiles_info[x_i].src_ey;
                    int sx    = tiles_info[x_i].src_sx;
                    int ex    = tiles_info[x_i].src_ex;
                    int count = (ex - sx) * 8;

                    // source transform start
                    auto src_start              = src_z + tiles_info[x_i].src_loc;
                    fp16_t *transform_dst       = dst_z + x_i * src_unit * src_unit * 8;
                    const fp16_t *transform_src = nullptr;

                    int h_stride0 = 0;

                    if (ex - sx == src_unit && ey - sy == src_unit) {
                        transform_src = src_start;
                        h_stride0     = 8 * k_param_->iw;
                    } else {
                        memset(mid_buffer, 0, src_unit * src_unit * 8 * data_byte_size);
                        if (count > 0) {
                            for (int yy = sy; yy < ey; yy++) {
                                auto dst_yy = mid_buffer + yy * src_unit * 8 + sx * 8;
                                auto src_yy = src_start + 8 * k_param_->iw * yy + sx * 8;
                                memcpy(dst_yy, src_yy, count * data_byte_size);
                            }
                        }

                        transform_src = mid_buffer;
                        h_stride0     = 8 * src_unit;
                    }

                    SrcTransformFunc_(transform_src, transform_dst, 8, h_stride0);
                    // source transform end
                }

                /*
                repack data format to nchw for gemm func
                total data num : ic * tile * unit * unit
                */
                if (src_unit == 4) {
                    load_repack_half<16>(repack_buf, dst_z, x_c, z, ic, k_param_->ic_r8);
                } else if (src_unit == 6) {
                    load_repack_half<36>(repack_buf, dst_z, x_c, z, ic, k_param_->ic_r8);
                }
            }

            // gemm multi (n8 for armv8, n4 for armv7)
            OMP_PARALLEL_FOR_
            for (int i = 0; i < src_unit * src_unit; i++) {
                GEMM_FP16_N8(dst_trans_buf + i * 8 * NEON_GEMM_TILE_HW,
                             repack_buf + i * k_param_->ic_r8 * NEON_GEMM_TILE_HW,
                             reinterpret_cast<fp16_t *>(k_param_->fil_ptr) + i * k_param_->ic_r8 * k_param_->oc_r8,
                             k_param_->ic_r8, NEON_GEMM_TILE_HW * src_unit * src_unit * 8, k_param_->oc_r8, x_c, fake_bias, 0);
            }

            src_z_step = NEON_GEMM_TILE_HW * src_unit * src_unit;
            dst_z_step = k_param_->ow * k_param_->oh;

            OMP_PARALLEL_FOR_
            for (int z = 0; z <= k_param_->oc_r8 - 8; z += 8) {
                int tid         = OMP_TID_;
                auto mid_buffer = src_pad_buffer + tid * src_pad_buf_per_thread;
                auto src_z      = dst_trans_buf + z * src_z_step;
                auto dst_z      = output_ptr + z * dst_z_step;
                for (int x_i = 0; x_i < x_c; x_i++) {
                    int ey = tiles_info[x_i].dst_ey;
                    int ex = tiles_info[x_i].dst_ex;

                    int count = ex * 8;
                    // dst transform start
                    fp16_t *transform_src = src_z + x_i * 8;
                    auto dst_start        = dst_z + tiles_info[x_i].dst_loc;
                    fp16_t *transform_dst = nullptr;
                    int h_stride0         = 8 * dst_unit;
                    int h_stride1         = 0;

                    if (ex == dst_unit) {
                        transform_dst = dst_start;
                        h_stride1     = 8 * k_param_->ow;
                    } else {
                        transform_dst = mid_buffer;
                        h_stride1     = 8 * dst_unit;
                    }

                    DstTransformFunc_(transform_src, transform_dst, NEON_GEMM_TILE_HW * 8, h_stride1, ey);

                    if (ex != dst_unit) {
                        for (int yy = 0; yy < ey; ++yy) {
                            auto dst_yy = dst_start + yy * 8 * k_param_->ow;
                            auto src_yy = mid_buffer + yy * 8 * dst_unit;
                            memcpy(dst_yy, src_yy, count * data_byte_size);
                        }
                    }
                    // dst transform end
                }
            }
        }
    }

    PostExec<fp16_t>(outputs);

    return TNN_OK;
}

}  // namespace TNN_NS
#endif
