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

#include "tnn/device/arm/acc/convolution/arm_conv_layer_3x3.h"

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

namespace TNN_NS {
/*
template transposition used in repack func
| a | b | c | d |      | a | e | i | m |
| e | f | g | h |  to  | b | f | g | h |
| i | j | k | l |      | c | g | k | o |
| m | n | o | p |      | d | h | l | p |
*/
template <typename T>
void transpose_4x4(T *v0, T *v1, T *v2, T *v3) {
    LOGE("TYPE NOT IMPLEMENT");
}

#ifdef TNN_USE_NEON
/*
template specialization for data type Float4(float&bfp16)
*/
template <>
void transpose_4x4(Float4 *v0, Float4 *v1, Float4 *v2, Float4 *v3) {
    float32x4x2_t q01 = vtrnq_f32(v0->value, v1->value);
    float32x4x2_t q23 = vtrnq_f32(v2->value, v3->value);

    float32x2_t d00 = vget_low_f32(q01.val[0]);
    float32x2_t d01 = vget_high_f32(q01.val[0]);

    float32x2_t d10 = vget_low_f32(q01.val[1]);
    float32x2_t d11 = vget_high_f32(q01.val[1]);

    float32x2_t d20 = vget_low_f32(q23.val[0]);
    float32x2_t d21 = vget_high_f32(q23.val[0]);

    float32x2_t d30 = vget_low_f32(q23.val[1]);
    float32x2_t d31 = vget_high_f32(q23.val[1]);

    v0->value = vcombine_f32(d00, d20);
    v1->value = vcombine_f32(d10, d30);
    v2->value = vcombine_f32(d01, d21);
    v3->value = vcombine_f32(d11, d31);
}
#else
/*
template specialization for data type Float4(float&bfp16)
*/
template <>
void transpose_4x4(Float4 *v0, Float4 *v1, Float4 *v2, Float4 *v3) {
    Float4 q0 = *v0;
    Float4 q1 = *v1;
    Float4 q2 = *v2;
    Float4 q3 = *v3;

    v0->value[0] = q0.value[0];
    v0->value[1] = q1.value[0];
    v0->value[2] = q2.value[0];
    v0->value[3] = q3.value[0];

    v1->value[0] = q0.value[1];
    v1->value[1] = q1.value[1];
    v1->value[2] = q2.value[1];
    v1->value[3] = q3.value[1];

    v2->value[0] = q0.value[2];
    v2->value[1] = q1.value[2];
    v2->value[2] = q2.value[2];
    v2->value[3] = q3.value[2];

    v3->value[0] = q0.value[3];
    v3->value[1] = q1.value[3];
    v3->value[2] = q2.value[3];
    v3->value[3] = q3.value[3];
}
#endif

/*
transpose 4x4
*/
template <typename T>
static inline void _repack_4(T *dst_b, T *src_b, int src_stride) {
    Float4 q0 = Float4::load(src_b + 0 * src_stride);
    Float4 q1 = Float4::load(src_b + 1 * src_stride);
    Float4 q2 = Float4::load(src_b + 2 * src_stride);
    Float4 q3 = Float4::load(src_b + 3 * src_stride);
    transpose_4x4(&q0, &q1, &q2, &q3);

    Float4::save(dst_b + 0, q0);
    Float4::save(dst_b + 4, q1);
    Float4::save(dst_b + 8, q2);
    Float4::save(dst_b + 12, q3);
}

/*
A0 & A1 are 4x4 matrix
| A0 | to | A0T | A1T |
| A1 |
*/
template <typename T>
static inline void _repack_8(T *dst_b, T *src_b, int src_stride) {
    Float4 q0 = Float4::load(src_b + 0 * src_stride);
    Float4 q1 = Float4::load(src_b + 1 * src_stride);
    Float4 q2 = Float4::load(src_b + 2 * src_stride);
    Float4 q3 = Float4::load(src_b + 3 * src_stride);
    transpose_4x4(&q0, &q1, &q2, &q3);

    Float4 q4 = Float4::load(src_b + 4 * src_stride);
    Float4 q5 = Float4::load(src_b + 5 * src_stride);
    Float4 q6 = Float4::load(src_b + 6 * src_stride);
    Float4 q7 = Float4::load(src_b + 7 * src_stride);
    transpose_4x4(&q4, &q5, &q6, &q7);

    Float4::save(dst_b + 0, q0);
    Float4::save(dst_b + 4, q4);
    Float4::save(dst_b + 8, q1);
    Float4::save(dst_b + 12, q5);
    Float4::save(dst_b + 16, q2);
    Float4::save(dst_b + 20, q6);
    Float4::save(dst_b + 24, q3);
    Float4::save(dst_b + 28, q7);
}

/*
A0 & A1 & A2 are 4x4 matrix
| A0 |
| A1 |  to | A0T | A1T | A2T |
| A2 |
*/
template <typename T>
static inline void _repack_12(T *dst_b, T *src_b, int src_stride) {
    Float4 q0 = Float4::load(src_b + 0 * src_stride);
    Float4 q1 = Float4::load(src_b + 1 * src_stride);
    Float4 q2 = Float4::load(src_b + 2 * src_stride);
    Float4 q3 = Float4::load(src_b + 3 * src_stride);
    transpose_4x4(&q0, &q1, &q2, &q3);

    Float4 q4 = Float4::load(src_b + 4 * src_stride);
    Float4 q5 = Float4::load(src_b + 5 * src_stride);
    Float4 q6 = Float4::load(src_b + 6 * src_stride);
    Float4 q7 = Float4::load(src_b + 7 * src_stride);
    transpose_4x4(&q4, &q5, &q6, &q7);

    Float4 q8  = Float4::load(src_b + 8 * src_stride);
    Float4 q9  = Float4::load(src_b + 9 * src_stride);
    Float4 q10 = Float4::load(src_b + 10 * src_stride);
    Float4 q11 = Float4::load(src_b + 11 * src_stride);
    transpose_4x4(&q8, &q9, &q10, &q11);

    Float4::save(dst_b + 0, q0);
    Float4::save(dst_b + 4, q4);
    Float4::save(dst_b + 8, q8);
    Float4::save(dst_b + 12, q1);
    Float4::save(dst_b + 16, q5);
    Float4::save(dst_b + 20, q9);
    Float4::save(dst_b + 24, q2);
    Float4::save(dst_b + 28, q6);
    Float4::save(dst_b + 32, q10);
    Float4::save(dst_b + 36, q3);
    Float4::save(dst_b + 40, q7);
    Float4::save(dst_b + 44, q11);
}

/*
load and pack data from nc4hw4 to nchw to maximize the efficiency of sgemm
*/
template <typename T>
static void load_repack(T *dst_b, T *src_b, int width, int src_stride) {
    if (width == ARM_SGEMM_TILE_M) {
#if defined(__aarch64__)
        _repack_12(dst_b, src_b, src_stride);
#else
        _repack_8(dst_b, src_b, src_stride);
#endif
    } else {
        int b_i = 0;
        for (; b_i + 3 < width; b_i += 4) {
            auto src_r = src_b + b_i * src_stride;
            auto dst_r = dst_b + b_i * 4;
            _repack_4(dst_r, src_r, src_stride);
        }
        for (; b_i < width; b_i++) {
            Float4::save(dst_b + b_i * 4, Float4::load(src_b + b_i * src_stride));
        }
    }
}

bool ArmConvLayer3x3::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                 const std::vector<Blob *> &outputs) {
    if (!param) {
        return false;
    }

    if (param->group != 1 || param->dialations[0] != 1 || param->dialations[1] != 1 || param->strides[0] != 1 ||
        param->kernels[0] != param->kernels[1] || param->strides[1] != 1 ||
        ROUND_UP(outputs[0]->GetBlobDesc().dims[1], 4) % ARM_SGEMM_TILE_N != 0) {
        return false;
    }

    if (!SelectWinograd(param, inputs, outputs)) {
        return false;
    }

    return true;
}

int ArmConvLayer3x3::SelectWinograd(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                    const std::vector<Blob *> &outputs) {
    if (!param) {
        return 0;
    }

    // gemm: only use gemm
    // winograd_unit2: use winograd f(2,3)
    // winograd_unit4 or others:
    // use winograd f(4,3) or f(2,3)
    if (param->extra_config.count("arm_fp32_gemm")) {
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
    float origin_cost = (float)ow * oh * (float)ROUND_UP(ic, 4) * ROUND_UP(oc, 4) * kernel_size * kernel_size;

    // only support F(2x2, 3x3) and F(4x4, 3x3)
    for (int u = 2; u <= 4; u += 2) {
        float src_unit = (float)(u + kernel_size - 1);

        // winograd cost = src transform + gemm + dst transform
        float winograd_cost =
            (2 * src_unit * src_unit * src_unit * ROUND_UP(ic, 4) +
             src_unit * src_unit * ROUND_UP(ic, 4) * ROUND_UP(oc, 4) + 2 * src_unit * u * u * ROUND_UP(oc, 4)) *
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

    if (param->extra_config.count("arm_fp32_winograd_unit2")) {
        dst_unit = 2;
    }

    return dst_unit;
}

ArmConvLayer3x3::~ArmConvLayer3x3() {}

Status ArmConvLayer3x3::allocateBufferWeight(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);
    ConvLayerResource *conv_res = dynamic_cast<ConvLayerResource *>(resource_);
    CHECK_PARAM_NULL(conv_res);

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;

    if (!buffer_weight_.GetBytesSize()) {
        const int input_channel  = dims_input[1];
        const int output_channel = dims_output[1];

        const int kw = conv_param->kernels[0];
        const int kh = conv_param->kernels[1];

        const int weight_bytes_count = conv_res->filter_handle.GetBytesSize();
        const float *src             = conv_res->filter_handle.force_to<float *>();
        int data_byte_size           = DataTypeUtils::GetBytesSize(conv_res->filter_handle.GetDataType());

        dst_unit_ = SelectWinograd(conv_param, inputs, outputs);
        src_unit_ = dst_unit_ + kw - 1;

        const int weight_count = src_unit_ * src_unit_ * k_param_->oc_r4 * k_param_->ic_r4;
        RawBuffer pack_weight(weight_count * data_byte_size + NEON_KERNEL_EXTRA_LOAD);

        switch (dst_unit_) {
            case 2:
                WeightTransform4x4(src, pack_weight.force_to<float *>(), 3, input_channel, output_channel);
                break;
            case 4:
                WeightTransform6x6(src, pack_weight.force_to<float *>(), 3, input_channel, output_channel);
                break;
            default:
                LOGE("Unsupport winograd dst unit\n");
                break;
        }

#ifdef __aarch64__
        for (int i = 0; i < src_unit_ * src_unit_; i++) {
            ConvertWeightsC4ToC8(pack_weight.force_to<float *>() + i * k_param_->ic_r4 * k_param_->oc_r4, dims_input[1],
                                 dims_output[1]);
        }
#endif
        buffer_weight_ = pack_weight;
    }

    return TNN_OK;
}

Status ArmConvLayer3x3::Init(Context *context, LayerParam *param, LayerResource *resource,
                             const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    RETURN_ON_NEQ(ArmConvLayerCommon::Init(context, param, resource, inputs, outputs), TNN_OK);
    auto in_data_type = inputs[0]->GetBlobDesc().data_type;

    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    if (conv_param) {
        if (in_data_type == DATA_TYPE_FLOAT) {
            if (dst_unit_ == 2) {
                SrcTransformFunc_ = SrcTransformInOne4x4Float;
                DstTransformFunc_ = DstTransformInOne4x2Float;
            } else if (dst_unit_ == 4) {
                SrcTransformFunc_ = SrcTransformInOne6x6Float;
                DstTransformFunc_ = DstTransformInOne6x4Float;
            } else {
                return TNNERR_LAYER_ERR;
            }
        } else if (in_data_type == DATA_TYPE_BFP16) {
            if (dst_unit_ == 2) {
                SrcTransformFunc_ = SrcTransformInOne4x4BFP16;
                DstTransformFunc_ = DstTransformInOne4x2BFP16;
            } else if (dst_unit_ == 4) {
                SrcTransformFunc_ = SrcTransformInOne6x6BFP16;
                DstTransformFunc_ = DstTransformInOne6x4BFP16;
            } else {
                return TNNERR_LAYER_ERR;
            }
        } else {
            return TNNERR_LAYER_ERR;
        }
    }
    return TNN_OK;
}

Status ArmConvLayer3x3::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto in_data_type = inputs[0]->GetBlobDesc().data_type;
    if (in_data_type == DATA_TYPE_FLOAT) {
        return Exec<float>(inputs, outputs);
    } else if (in_data_type == DATA_TYPE_BFP16) {
        return Exec<bfp16_t>(inputs, outputs);
    } else {
        return TNNERR_LAYER_ERR;
    }
}

template <typename T>
Status ArmConvLayer3x3::Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    auto input                 = inputs[0];
    auto output                = outputs[0];

    DataType data_type = output->GetBlobDesc().data_type;
    int data_byte_size = DataTypeUtils::GetBytesSize(data_type);

    const int batch = output->GetBlobDesc().dims[0];

    auto w_unit      = UP_DIV(k_param_->ow, dst_unit_);
    auto h_unit      = UP_DIV(k_param_->oh, dst_unit_);
    auto title_count = UP_DIV(w_unit * h_unit, ARM_SGEMM_TILE_M);

    T *src_origin = reinterpret_cast<T *>(GetBlobHandlePtr(input->GetHandle()));
    T *dst_origin = reinterpret_cast<T *>(GetBlobHandlePtr(output->GetHandle()));

    int max_num_threads          = OMP_MAX_THREADS_NUM_;
    int transform_num_per_thread = src_unit_ * src_unit_ * 4;
    int work_num_per_thread      = (k_param_->ic_r4 * 2 + k_param_->oc_r4) * src_unit_ * src_unit_ * ARM_SGEMM_TILE_M;

    auto tranform_buf_size = max_num_threads * transform_num_per_thread * sizeof(float);
    auto work_buf_size     = work_num_per_thread * sizeof(float);

    // gemm kernel need bias pointer
    auto fake_bias_size = k_param_->oc_r4 * sizeof(float);
    float *work_sapce   = reinterpret_cast<float *>(
        context_->GetSharedWorkSpace(tranform_buf_size + work_buf_size + fake_bias_size + NEON_KERNEL_EXTRA_LOAD));
    float *fake_bias    = reinterpret_cast<float *>(work_sapce);
    T *transform_buffer = reinterpret_cast<T *>(work_sapce + fake_bias_size / sizeof(float));
    work_sapce += tranform_buf_size / sizeof(float) + fake_bias_size / sizeof(float);

    // memset fake bias data to get correct results
    memset(fake_bias, 0, fake_bias_size);

    if (DstTransformFunc_ == nullptr || SrcTransformFunc_ == nullptr) {
        return TNNERR_COMMON_ERROR;
    }

    for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
        auto input_ptr  = src_origin + batch_idx * k_param_->iw * k_param_->ih * k_param_->ic_r4;
        auto output_ptr = dst_origin + batch_idx * k_param_->ow * k_param_->oh * k_param_->oc_r4;

        for (int t_idx = 0; t_idx < title_count; t_idx++) {
            auto _src_origin = work_sapce;
            auto _dst_origin = _src_origin + k_param_->ic_r4 * src_unit_ * src_unit_ * ARM_SGEMM_TILE_M;
            auto repack_buf  = _dst_origin + k_param_->oc_r4 * src_unit_ * src_unit_ * ARM_SGEMM_TILE_M;

            int x_idx    = t_idx * ARM_SGEMM_TILE_M;
            int x_remain = w_unit * h_unit - x_idx;
            int x_c      = x_remain > ARM_SGEMM_TILE_M ? ARM_SGEMM_TILE_M : x_remain;

            int src_z_step = k_param_->iw * k_param_->ih * 4;
            int dst_z_step = x_c * src_unit_ * src_unit_ * 4;

            OMP_PARALLEL_FOR_
            for (int z = 0; z < k_param_->ic_r4 / 4; z++) {
                int tid         = OMP_TID_;
                auto mid_buffer = transform_buffer + tid * transform_num_per_thread;
                auto src_z      = input_ptr + z * src_z_step;
                auto dst_z      = _src_origin + z * dst_z_step;
                for (int x_i = 0; x_i < x_c; x_i++) {
                    int idx   = x_idx + x_i;
                    int w_idx = idx % w_unit;
                    int h_idx = idx / w_unit;

                    int src_x = w_idx * dst_unit_ - conv_param->pads[0];
                    int src_y = h_idx * dst_unit_ - conv_param->pads[2];
                    int sy    = MAX(0, src_y) - src_y;
                    int ey    = MIN(src_y + src_unit_, k_param_->ih) - src_y;
                    int sx    = MAX(0, src_x) - src_x;
                    int ex    = MIN(src_x + src_unit_, k_param_->iw) - src_x;
                    int count = (ex - sx) * 4;

                    // source transform start
                    auto src_start       = src_z + (src_x + src_y * k_param_->iw) * 4;
                    auto dst_start       = dst_z + x_i * src_unit_ * src_unit_ * 4;
                    T *transform_src     = nullptr;
                    float *transform_dst = dst_start;
                    int h_stride0        = 0;
                    int h_stride1        = 4 * src_unit_;

                    if (ex - sx == src_unit_ && ey - sy == src_unit_) {
                        transform_src = src_start;
                        h_stride0     = 4 * k_param_->iw;
                    } else {
                        memset(mid_buffer, 0, src_unit_ * src_unit_ * 4 * data_byte_size);
                        if (count > 0) {
                            for (int yy = sy; yy < ey; yy++) {
                                auto dst_yy = mid_buffer + yy * src_unit_ * 4 + sx * 4;
                                auto src_yy = src_start + 4 * k_param_->iw * yy + sx * 4;
                                memcpy(dst_yy, src_yy, count * data_byte_size);
                            }
                        }

                        transform_src = mid_buffer;
                        h_stride0     = 4 * src_unit_;
                    }

                    SrcTransformFunc_(transform_src, transform_dst, 4, h_stride0);
                    // source transform end
                }

                /*
                repack data format to nchw for gemm func
                total data num : ic * tile * unit * unit
                */
                auto repack_z = repack_buf + 4 * x_c * z;
                for (int i = 0; i < src_unit_ * src_unit_; i++) {
                    auto repack_dst = repack_z + i * x_c * k_param_->ic_r4;
                    auto repack_src = dst_z + i * 4;
                    load_repack(repack_dst, repack_src, x_c, src_unit_ * src_unit_ * 4);
                }
            }

            // gemm multi (n8 for armv8, n4 for armv7)
            OMP_PARALLEL_FOR_
            for (int i = 0; i < src_unit_ * src_unit_; i++) {
                GEMM_FUNC(_dst_origin + i * 4 * x_c, repack_buf + i * k_param_->ic_r4 * x_c,
                          reinterpret_cast<float *>(k_param_->fil_ptr) + i * k_param_->ic_r4 * k_param_->oc_r4,
                          k_param_->ic_r4 / 4, x_c * src_unit_ * src_unit_ * 4, k_param_->oc_r4 / 4, x_c, fake_bias, 0);
            }

            src_z_step = x_c * src_unit_ * src_unit_ * 4;
            dst_z_step = k_param_->ow * k_param_->oh * 4;

            OMP_PARALLEL_FOR_
            for (int z = 0; z < k_param_->oc_r4 / 4; z++) {
                int tid         = OMP_TID_;
                auto mid_buffer = transform_buffer + tid * transform_num_per_thread;
                auto src_z      = _dst_origin + z * src_z_step;
                auto dst_z      = output_ptr + z * dst_z_step;
                for (int x_i = 0; x_i < x_c; x_i++) {
                    int idx   = x_idx + x_i;
                    int w_idx = idx % w_unit;
                    int h_idx = idx / w_unit;

                    int dst_x = w_idx * dst_unit_;
                    int dst_y = h_idx * dst_unit_;
                    int ey    = MIN(dst_y + dst_unit_, k_param_->oh) - dst_y;
                    int ex    = MIN(dst_x + dst_unit_, k_param_->ow) - dst_x;

                    int count = ex * 4;
                    // dst transform start
                    auto src_start       = src_z + x_i * 4;
                    auto dst_start       = dst_z + 4 * (dst_x + dst_y * k_param_->ow);
                    float *transform_src = src_start;
                    T *transform_dst     = nullptr;
                    int h_stride0        = 4 * dst_unit_;
                    int h_stride1        = 0;

                    if (ex == dst_unit_) {
                        transform_dst = dst_start;
                        h_stride1     = 4 * k_param_->ow;
                    } else {
                        transform_dst = mid_buffer;
                        h_stride1     = 4 * dst_unit_;
                    }

                    DstTransformFunc_(transform_src, transform_dst, x_c * 4, h_stride1, ey);

                    if (ex != dst_unit_) {
                        for (int yy = 0; yy < ey; ++yy) {
                            auto dst_yy = dst_start + yy * 4 * k_param_->ow;
                            auto src_yy = mid_buffer + yy * 4 * dst_unit_;
                            memcpy(dst_yy, src_yy, count * data_byte_size);
                        }
                    }
                    // dst transform end
                }
            }
        }
    }

    PostExec<T>(outputs);

    return TNN_OK;
}

}  // namespace TNN_NS
