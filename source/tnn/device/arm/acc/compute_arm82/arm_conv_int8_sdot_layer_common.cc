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

#include "tnn/device/arm/acc/convolution/arm_conv_int8_sdot_layer_common.h"
#include "tnn/device/arm/acc/compute_arm82/compute_sdot_int8.h"

#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/omp_utils.h"
#include "tnn/utils/cpu_utils.h"

#ifdef TNN_ARM82_USE_NEON

namespace TNN_NS {

bool ArmConvInt8SdotLayerCommon::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs) {
    bool support_dot = CpuUtils::CpuSupportInt8Dot();
    if (support_dot && inputs[0]->GetBlobDesc().data_type == DATA_TYPE_INT8) {
        return true;
    }
    return false;
}

ArmConvInt8SdotLayerCommon::~ArmConvInt8SdotLayerCommon() {}

Status ArmConvInt8SdotLayerCommon::allocateBufferWeight(const std::vector<Blob *> &inputs,
                                                    const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);
    ConvLayerResource *conv_res = dynamic_cast<ConvLayerResource *>(resource_);
    CHECK_PARAM_NULL(conv_res);

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;

    if (!buffer_weight_.GetBytesSize()) {
        int kw = conv_param->kernels[0];
        int kh = conv_param->kernels[1];

        int oc     = dims_output[1];
        int ic     = dims_input[1];
        int oc_r4  = ROUND_UP(oc, 4);
        int crs_r4 = ROUND_UP(ic, 4) * kw * kh;

        int weight_count     = oc_r4 * crs_r4;
        int weight_byte_size = weight_count * DataTypeUtils::GetBytesSize(conv_res->filter_handle.GetDataType());
        RawBuffer temp_buffer(weight_byte_size + NEON_KERNEL_EXTRA_LOAD);

        auto weight_src = conv_res->filter_handle.force_to<int8_t *>();
        // temp_buffer has been memset to 0
        auto weight_dst = temp_buffer.force_to<int8_t *>();
        PackSDOTINT8Weight(weight_src, weight_dst, oc, ic, kh, kw);

        buffer_weight_ = temp_buffer;
    }

    return TNN_OK;
}

// aarch32 memcpy small size has poor performance, use intrinsic to speed up
static inline void memcpy_intrinsic(int8_t *dst, const int8_t *src, int ic_r4) {
    int i = 0;
    for (; i + 31 < ic_r4; i += 32) {
        vst1q_s8(dst + i, vld1q_s8(src + i));
        vst1q_s8(dst + i + 16, vld1q_s8(src + i + 16));
    }
    for (; i + 15 < ic_r4; i += 16) {
        vst1q_s8(dst + i, vld1q_s8(src + i));
    }
    for (; i + 7 < ic_r4; i += 8) {
        vst1_s8(dst + i, vld1_s8(src + i));
    }
    for (; i + 3 < ic_r4; i += 4) {
        *((int32_t*)(dst + i)) = *((int32_t*)(src + i));
    }
}

#define DEF_IMG2COL_VAL                                                                                                \
    int x_id = (int)x_start + i;                                                                                       \
    int ox   = x_id % output_dims[3];                                                                                  \
    int oy   = x_id / output_dims[3];                                                                                  \
    int sx   = ox * param->strides[0] - param->pads[0];                                                                \
    int sy   = oy * param->strides[1] - param->pads[2];                                                                \
    int sfy  = MAX(0, (UP_DIV(-sy, param->dialations[1])));                                                            \
    int efy  = MIN(kh, UP_DIV(input_dims[2] - sy, param->dialations[1]));                                              \
    int sfx  = MAX(0, (UP_DIV(-sx, param->dialations[0])));                                                            \
    int efx  = MIN(kw, UP_DIV(input_dims[3] - sx, param->dialations[0]));                                              \
    int fyC  = efy - sfy;                                                                                              \
    int fxC  = efx - sfx;

/*
general img2col func
*/
static void im2col(int8_t *dst, const int8_t *src, const ConvLayerParam *param, size_t x_start, size_t dst_cnt,
                   int crs_r4, DimsVector input_dims, DimsVector output_dims) {
    const int src_w_step = ROUND_UP(input_dims[1], 4);
    auto kh       = param->kernels[1];
    auto kw       = param->kernels[0];
    auto dilate_y = param->dialations[1];
    auto dilate_x = param->dialations[0];
    for (int i = 0; i < dst_cnt; ++i) {
        DEF_IMG2COL_VAL;

        auto dst_i        = dst + crs_r4 * i;
        auto input_offset = src + (sx + sfx * dilate_x + (sy + sfy * dilate_y) * input_dims[3]) * src_w_step;
        auto idx_offset   = (sfy * kw + sfx) * src_w_step;
        memset(dst_i, 0, crs_r4 * sizeof(int8_t));
        for (int fy = 0; fy < fyC; ++fy) {
            auto dst_y = dst_i + idx_offset + fy * kw * src_w_step;
            auto src_y = input_offset + fy * input_dims[3] * src_w_step * dilate_y;
            for (int fx = 0; fx < fxC; ++fx) {
                auto dst_x = dst_y + fx * src_w_step;
                auto src_x = src_y + fx * dilate_x * src_w_step;
                memcpy_intrinsic(dst_x, src_x, src_w_step);
            }
        }
    }
}

// hard code src_w_step to 4 can speed up aarch32
static void im2col_smallc(int8_t *dst, const int8_t *src, const ConvLayerParam *param, size_t x_start, size_t dst_cnt,
                          int crs_r4, DimsVector input_dims, DimsVector output_dims) {
    const int src_w_step = 4; // ROUND_UP(input_dims[1], 4)
    auto kh       = param->kernels[1];
    auto kw       = param->kernels[0];
    auto dilate_y = param->dialations[1];
    auto dilate_x = param->dialations[0];
    for (int i = 0; i < dst_cnt; ++i) {
        DEF_IMG2COL_VAL;

        auto dst_i        = dst + crs_r4 * i;
        auto input_offset = src + (sx + sfx * dilate_x + (sy + sfy * dilate_y) * input_dims[3]) * src_w_step;
        auto idx_offset   = (sfy * kw + sfx) * src_w_step;
        memset(dst_i, 0, crs_r4 * sizeof(int8_t));
        for (int fy = 0; fy < fyC; ++fy) {
            auto dst_y = dst_i + idx_offset + fy * kw * src_w_step;
            auto src_y = input_offset + fy * input_dims[3] * src_w_step * dilate_y;
            for (int fx = 0; fx < fxC; ++fx) {
                auto dst_x = dst_y + fx * src_w_step;
                auto src_x = src_y + fx * dilate_x * src_w_step;
                memcpy(dst_x, src_x, src_w_step);
            }
        }
    }
}

Status ArmConvInt8SdotLayerCommon::Init(Context *context, LayerParam *param, LayerResource *resource,
                                    const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    RETURN_ON_NEQ(ArmLayerAcc::Init(context, param, resource, inputs, outputs), TNN_OK);
    RETURN_ON_NEQ(allocateBufferBias(inputs, outputs), TNN_OK);
    RETURN_ON_NEQ(allocateBufferScale(inputs, outputs), TNN_OK);
    RETURN_ON_NEQ(allocateBufferWeight(inputs, outputs), TNN_OK);
    RETURN_ON_NEQ(setFusionParam(inputs, outputs), TNN_OK);

    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);

    auto dims_input = inputs[0]->GetBlobDesc().dims;
    int kernel_x    = conv_param->kernels[0];
    int kernel_y    = conv_param->kernels[1];
    int stride_x    = conv_param->strides[0];
    int stride_y    = conv_param->strides[1];
    int pad_x       = conv_param->pads[0];
    int pad_y       = conv_param->pads[2];
    int ic          = dims_input[1];
    int ic_r4       = ROUND_UP(ic, 4);

    // fast mode
    bool no_im2col = kernel_x == 1 && kernel_y == 1 && ic_r4 % 4 == 0 && stride_x == 1 && stride_y == 1 &&
                     pad_x == 0 && pad_y == 0;
    if (!no_im2col) {
        im_col_func_ = im2col;
        if (dims_input[1] <= 4) {
            im_col_func_ = im2col_smallc;
        }
    } else {
        im_col_func_ = nullptr;
    }

    // set tile blk size, which be limit to 16KB
    // 16 * 1024 / sizeof(int8_t)
    int tile_blk = 16384 / (ic_r4 * kernel_x * kernel_y);
    tile_blk = ROUND_UP(tile_blk, NEON_INT8_SDOT_TILE_HW);
    if (tile_blk < NEON_INT8_SDOT_TILE_HW) {
        tile_blk = NEON_INT8_SDOT_TILE_HW;
    }
    if (tile_blk > 1024) {
        tile_blk = 1024;
    }
    tile_blk_ = tile_blk;

    return TNN_OK;
}

Status ArmConvInt8SdotLayerCommon::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);
    auto input     = inputs[0];
    auto output    = outputs[0];
    auto add_input = (conv_param->fusion_type == FusionType_None) ? nullptr : inputs[1];

    DataType data_type = output->GetBlobDesc().data_type;
    int data_byte_size = DataTypeUtils::GetBytesSize(data_type);

    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;
    const int batch  = dims_output[0];
    auto ic          = dims_input[1];
    auto ic_r4       = ROUND_UP(ic, 4);
    auto oc_r4       = ROUND_UP(dims_output[1], 4);
    auto oc_r4_align = oc_r4 / 8 * 8;

    auto input_channel_stride  = DimsVectorUtils::Count(dims_input, 2);
    auto output_channel_stride = DimsVectorUtils::Count(dims_output, 2);
    auto input_batch_stride    = input_channel_stride * ic_r4;
    auto output_batch_stride   = output_channel_stride * oc_r4;

    int8_t *input_data     = reinterpret_cast<int8_t *>(GetBlobHandlePtr(input->GetHandle()));
    int8_t *output_data    = reinterpret_cast<int8_t *>(GetBlobHandlePtr(output->GetHandle()));
    int8_t *add_input_data = add_input ? reinterpret_cast<int8_t *>(GetBlobHandlePtr(add_input->GetHandle())) : nullptr;

    float *scale_ptr      = buffer_scale_.force_to<float *>();
    int32_t *bias_ptr     = buffer_bias_.force_to<int32_t *>();
    int8_t *weight_ptr    = buffer_weight_.force_to<int8_t *>();
    float *add_scale_ptr  = buffer_add_scale_.force_to<float *>();
    int8_t *relu6_max_ptr = relu6_max_.force_to<int8_t *>();

    const int crs_r4 = ic_r4 * conv_param->kernels[1] * conv_param->kernels[0];
    int tile_count = UP_DIV(dims_output[2] * dims_output[3], tile_blk_);

    // for multi-threads, adjust tile_blk to make more threads parallel
    int max_num_threads = OMP_MAX_THREADS_NUM_;
    if (max_num_threads > 1) {
        while (tile_count < max_num_threads && tile_blk_ > NEON_INT8_SDOT_TILE_HW) {
            tile_blk_ = ROUND_UP(tile_blk_ / 2, NEON_INT8_SDOT_TILE_HW);
            tile_count = UP_DIV(dims_output[2] * dims_output[3], tile_blk_);
        }
    }

    size_t workspace_size = 0;
    size_t im2col_buf_size = crs_r4 * tile_blk_ * max_num_threads + NEON_KERNEL_EXTRA_LOAD;
    int8_t *workspace = reinterpret_cast<int8_t *>(context_->GetSharedWorkSpace(im2col_buf_size));
    auto im2col_buf_ptr = workspace;

#if defined(TNN_ARM82_A64)
    auto GemmInt8SdotUnit = GemmInt8SdotUnit8x8;
    auto GemmInt8SdotLeft = GemmInt8SdotUnit8x4;
#elif defined(TNN_ARM82_A32)
    auto GemmInt8SdotUnit = GemmInt8SdotUnit4x8;
    auto GemmInt8SdotLeft = GemmInt8SdotUnit4x4;
#endif

    for (int n = 0; n < batch; ++n) {
        auto input_batch     = input_data + n * input_batch_stride;
        auto output_batch    = output_data + n * output_batch_stride;
        auto add_input_batch = add_input_data ? add_input_data + n * output_batch_stride : nullptr;

        OMP_PARALLEL_FOR_GUIDED_
        for (int t_idx = 0; t_idx < tile_count; t_idx++) {
            int thread_id          = OMP_TID_;
            int8_t *input_kernel   = nullptr;
            const int hw_start     = t_idx * tile_blk_;
            const int real_hw_tile = MIN(output_channel_stride - hw_start, tile_blk_);
            const int input_count  = crs_r4 * tile_blk_;

            // im2col
            if (im_col_func_) {
                input_kernel = im2col_buf_ptr + input_count * thread_id;
                im_col_func_(input_kernel, input_batch, conv_param,
                                hw_start, real_hw_tile, crs_r4, dims_input, dims_output);
            } else {
                input_kernel = input_batch + hw_start * ic_r4;
            }
            auto output_kernel = output_batch + hw_start * oc_r4;
            auto add_input_kernel = add_input_batch ? add_input_batch + hw_start * oc_r4 : nullptr;

            GemmInt8SdotUnit(output_kernel, input_kernel, weight_ptr, crs_r4, oc_r4, real_hw_tile,
                                bias_ptr, scale_ptr, relu_, add_input_kernel, add_scale_ptr, relu6_max_ptr);

            if (oc_r4 > oc_r4_align) {
                auto add_input_tmp = add_input_kernel ? add_input_kernel + oc_r4_align : nullptr;
                GemmInt8SdotLeft(output_kernel + oc_r4_align, input_kernel,
                                   weight_ptr + oc_r4_align * crs_r4,
                                   crs_r4, oc_r4, real_hw_tile,
                                   bias_ptr + oc_r4_align, scale_ptr + oc_r4_align,
                                   relu_, add_input_tmp, add_scale_ptr + oc_r4_align,
                                   relu6_max_ptr + oc_r4_align);
            }
        }
    }

    return TNN_OK;
}

}  // namespace TNN_NS
#endif
#endif
