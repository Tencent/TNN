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

#include "tnn/device/x86/acc/convolution/x86_conv_int8_layer_common.h"

#include "tnn/device/x86/x86_common.h"
#include "tnn/device/x86/x86_context.h"
#include "tnn/device/x86/acc/compute/x86_compute_int8.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/omp_utils.h"
#include "tnn/utils/naive_compute.h"

namespace TNN_NS {
using namespace x86;

/*
X86ConvInt8LayerCommon as the last conv int8 solution
*/
bool X86ConvInt8LayerCommon::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs) {
    if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_INT8) {
        return true;
    }
    return false;
}

X86ConvInt8LayerCommon::~X86ConvInt8LayerCommon() {}

Status X86ConvInt8LayerCommon::allocateBufferWeight(const std::vector<Blob *> &inputs,
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
        const int group = conv_param->group;

        const int oc         = dims_output[1];
        const int ic         = dims_input[1];
        const int oc_g       = oc / group;
        const int ic_g       = ic / group;
        const int oc_g_r4    = ROUND_UP(oc_g, 4);
        const int ic_g_r4    = ROUND_UP(ic_g, 4);
        const int icrs_g_r16 = ROUND_UP(ic_g_r4 * kw * kh, 16);
        const int icrs_g     = ic_g * kw * kh;

        int weight_count   = group * oc_g_r4 * icrs_g_r16;
        int data_byte_size = weight_count * DataTypeUtils::GetBytesSize(conv_res->filter_handle.GetDataType());
        RawBuffer temp_buffer(data_byte_size + SIMD_KERNEL_EXTRA_LOAD);

        for (int g = 0; g < group; g++) {
            auto weight_src_g = conv_res->filter_handle.force_to<int8_t *>() + g * oc_g * icrs_g;
            auto weight_dst_g = temp_buffer.force_to<int8_t *>() + g * oc_g_r4 * icrs_g_r16;
            // from [o][i][h][w]
            // to: [o/4][h][w][i/16][o4][i16]
            PackINT8Weight(weight_src_g, weight_dst_g, ic_g, oc_g,
                           conv_param->kernels[1], conv_param->kernels[0]);
        }
        buffer_weight_ = temp_buffer;
    }
    return TNN_OK;
}

Status X86ConvInt8LayerCommon::allocateBufferBias(const std::vector<Blob *> &inputs,
                                                  const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);
    ConvLayerResource *conv_res = dynamic_cast<ConvLayerResource *>(resource_);
    CHECK_PARAM_NULL(conv_res);

    auto dims_output = outputs[0]->GetBlobDesc().dims;
    if (!buffer_bias_.GetBytesSize()) {
        if (conv_param->bias) {
            int total_byte_size =
                ROUND_UP(dims_output[1], 4) * DataTypeUtils::GetBytesSize(conv_res->bias_handle.GetDataType());

            const int bias_handle_size      = conv_res->bias_handle.GetBytesSize();
            const int32_t *bias_handle_data = conv_res->bias_handle.force_to<int32_t *>();

            RawBuffer temp_buffer(total_byte_size);
            memcpy(temp_buffer.force_to<int32_t *>(), conv_res->bias_handle.force_to<int32_t *>(), bias_handle_size);

            buffer_bias_ = temp_buffer;
        } else if (outputs[0]->GetBlobDesc().data_type == DATA_TYPE_INT8) {
            // int 8 kernel always add bias, if not, set zeros
            buffer_bias_ = RawBuffer(ROUND_UP(dims_output[1], 4) * sizeof(int32_t));
        }
    }

    return TNN_OK;
}

Status X86ConvInt8LayerCommon::allocateBufferScale(const std::vector<Blob *> &inputs,
                                                   const std::vector<Blob *> &outputs) {
    ConvLayerResource *conv_res = dynamic_cast<ConvLayerResource *>(resource_);
    CHECK_PARAM_NULL(conv_res);

    // alloc scale buffer
    if (!buffer_scale_.GetBytesSize()) {
        auto dims_output = outputs[0]->GetBlobDesc().dims;
        int total_byte_size =
            ROUND_UP(dims_output[1], 4) * DataTypeUtils::GetBytesSize(conv_res->scale_handle.GetDataType());

        const int scale_handle_size = conv_res->scale_handle.GetBytesSize();
        const float *w_scale        = conv_res->scale_handle.force_to<float *>();

        const float *o_scale =
            reinterpret_cast<BlobInt8 *>(outputs[0])->GetIntResource()->scale_handle.force_to<float *>();

        int scale_len_w = conv_res->scale_handle.GetDataCount();
        int scale_len_o = reinterpret_cast<BlobInt8 *>(outputs[0])->GetIntResource()->scale_handle.GetDataCount();
        RawBuffer temp_buffer(total_byte_size);
        float *temp_ptr = temp_buffer.force_to<float *>();
        for (int i = 0; i < dims_output[1]; i++) {
            int scale_idx_w = scale_len_w == 1 ? 0 : i;
            int scale_idx_o = scale_len_o == 1 ? 0 : i;

            if (w_scale[scale_idx_w] < 0.0f || o_scale[scale_idx_o] < 0.0f) {
                return Status(TNNERR_PARAM_ERR, "int8-blob scale can not be negative");
            }

            if (o_scale[scale_idx_o] >= FLT_MIN)
                temp_ptr[i] = w_scale[scale_idx_w] / o_scale[scale_idx_o];
            else
                temp_ptr[i] = 0.0;
        }
        buffer_scale_ = temp_buffer;
    }

    return TNN_OK;
}

Status X86ConvInt8LayerCommon::allocateBufferAddScale(const std::vector<Blob *> &inputs,
                                                      const std::vector<Blob *> &outputs) {
    ConvLayerResource *conv_res = dynamic_cast<ConvLayerResource *>(resource_);
    CHECK_PARAM_NULL(conv_res);

    if (DimsVectorUtils::Count(inputs[1]->GetBlobDesc().dims) !=
        DimsVectorUtils::Count(outputs[0]->GetBlobDesc().dims)) {
        return Status(TNNERR_LAYER_ERR, "Conv-Add fusion does not support broadcast-add");
    }

    // alloc add scale buffer
    if (!buffer_add_scale_.GetBytesSize()) {
        auto dims_output = outputs[0]->GetBlobDesc().dims;
        int total_byte_size =
            ROUND_UP(dims_output[1], 4) * DataTypeUtils::GetBytesSize(conv_res->scale_handle.GetDataType());

        const float *i_scale =
            reinterpret_cast<BlobInt8 *>(inputs[1])->GetIntResource()->scale_handle.force_to<float *>();

        const float *o_scale =
            reinterpret_cast<BlobInt8 *>(outputs[0])->GetIntResource()->scale_handle.force_to<float *>();

        int scale_len_i = reinterpret_cast<BlobInt8 *>(inputs[1])->GetIntResource()->scale_handle.GetDataCount();
        int scale_len_o = reinterpret_cast<BlobInt8 *>(outputs[0])->GetIntResource()->scale_handle.GetDataCount();
        RawBuffer temp_buffer(total_byte_size);
        float *temp_ptr = temp_buffer.force_to<float *>();
        for (int i = 0; i < dims_output[1]; i++) {
            int scale_idx_i = scale_len_i == 1 ? 0 : i;
            int scale_idx_o = scale_len_o == 1 ? 0 : i;

            if (i_scale[scale_idx_i] < 0.0f || o_scale[scale_idx_o] < 0.0f) {
                return Status(TNNERR_PARAM_ERR, "int8-blob scale can not be negative");
            }

            if (o_scale[scale_idx_o] >= FLT_MIN)
                temp_ptr[i] = i_scale[scale_idx_i] / o_scale[scale_idx_o];
            else
                temp_ptr[i] = 0.0;
        }
        buffer_add_scale_ = temp_buffer;
    }

    return TNN_OK;
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
                   int crs_div8, DimsVector input_dims, DimsVector output_dims) {
    const int src_w_step      = ROUND_UP(input_dims[1], 4);
    const int crs_r8          = crs_div8 * 8;
    auto kh       = param->kernels[1];
    auto kw       = param->kernels[0];
    auto dilate_y = param->dialations[1];
    auto dilate_x = param->dialations[0];
    for (int i = 0; i < dst_cnt; ++i) {
        DEF_IMG2COL_VAL;

        auto dst_i        = dst + crs_r8 * i;
        auto input_offset = src + (sx + sfx * dilate_x + (sy + sfy * dilate_y) * input_dims[3]) * src_w_step;
        auto idx_offset   = (sfy * kw + sfx) * src_w_step;
        memset(dst_i, 0, crs_r8 * sizeof(int8_t));
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
    if (ROUND_UP(dst_cnt, 4) > dst_cnt) {
        memset(dst + crs_r8 * dst_cnt, 0, crs_r8 * (ROUND_UP(dst_cnt, 4) - dst_cnt) * sizeof(int8_t));
    }
}

/*
template img2col func when c is small(eg. 1,2,3)
*/
template <int REALC>
static void im2col_smallc(int8_t *dst, const int8_t *src, const ConvLayerParam *param, size_t x_start, size_t dst_cnt,
                          int crs_div8, DimsVector input_dims, DimsVector output_dims) {
    const int src_w_step      = 4;
    const int crs_r8          = crs_div8 * 8;
    auto kh       = param->kernels[1];
    auto kw       = param->kernels[0];
    auto dilate_y = param->dialations[1];
    auto dilate_x = param->dialations[0];
    for (int i = 0; i < dst_cnt; ++i) {
        DEF_IMG2COL_VAL;

        auto dst_i        = dst + crs_r8 * i;
        auto input_offset = src + (sx + sfx * dilate_x + (sy + sfy * dilate_y) * input_dims[3]) * src_w_step;
        auto idx_offset   = (sfy * kw + sfx) * REALC;
        memset(dst_i, 0, crs_r8 * sizeof(int8_t));
        for (int fy = 0; fy < fyC; ++fy) {
            auto dst_y = dst_i + idx_offset + fy * kw * REALC;
            auto src_y = input_offset + fy * input_dims[3] * src_w_step * dilate_y;
            for (int fx = 0; fx < fxC; fx++) {
                auto dst_x = dst_y + fx * REALC;
                auto src_x = src_y + fx * src_w_step * dilate_x;
                for (int c = 0; c < REALC; c++) {
                    dst_x[c] = src_x[c];
                }
            }
        }
    }
    if (ROUND_UP(dst_cnt, 4) > dst_cnt) {
        memset(dst + crs_r8 * dst_cnt, 0, crs_r8 * (ROUND_UP(dst_cnt, 4) - dst_cnt) * sizeof(int8_t));
    }
}

Status X86ConvInt8LayerCommon::setFusionParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);

    // fused add input
    if (conv_param->fusion_type != FusionType_None) {
        RETURN_ON_NEQ(allocateBufferAddScale(inputs, outputs), TNN_OK);
    }

    // only support relu activation
    if (conv_param->activation_type == ActivationType_ReLU) {
        relu_ = 1;
        if (conv_param->fusion_type == FusionType_Conv_Activation_Add) {
            relu_ = -1;
        }
    } else if (conv_param->activation_type == ActivationType_ReLU6) {
        relu_ = 2;
        if (conv_param->fusion_type == FusionType_Conv_Activation_Add) {
            return Status(TNNERR_LAYER_ERR, "Conv-Activation-Add fusion does not support relu6");
        }
    }

    // compute relu6 max
    if (conv_param->activation_type == ActivationType_ReLU6) {
        auto output_scale_resource      = reinterpret_cast<BlobInt8 *>(outputs[0])->GetIntResource();
        auto output_scale_len           = output_scale_resource->scale_handle.GetDataCount();
        auto output_scale_resource_data = output_scale_resource->scale_handle.force_to<float *>();
        auto &dims_output               = outputs[0]->GetBlobDesc().dims;
        auto &output_channel            = dims_output[1];
        RawBuffer relu6_max             = RawBuffer(ROUND_UP(output_channel, 4) * sizeof(int8_t));
        auto relu6_max_data             = relu6_max.force_to<int8_t *>();
        for (int i = 0; i < output_channel; ++i) {
            int scale_idx     = output_scale_len == 1 ? 0 : i;
            relu6_max_data[i] = float2int8(6.0f / output_scale_resource_data[scale_idx]);
        }
        for (int i = output_channel; i < ROUND_UP(output_channel, 4); ++i) {
            relu6_max_data[i] = 127;
        }
        relu6_max_ = relu6_max;
        relu6_max_.SetDataType(DATA_TYPE_INT8);
    }

    return TNN_OK;
}

Status X86ConvInt8LayerCommon::Init(Context *context, LayerParam *param, LayerResource *resource,
                                    const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    RETURN_ON_NEQ(X86LayerAcc::Init(context, param, resource, inputs, outputs), TNN_OK);
    RETURN_ON_NEQ(allocateBufferBias(inputs, outputs), TNN_OK);
    RETURN_ON_NEQ(allocateBufferScale(inputs, outputs), TNN_OK);
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
    int ic_g        = dims_input[1] / conv_param->group;
    int ic_g_r4     = ROUND_UP(ic_g, 4);

    // fast mode
    bool no_im2col = kernel_x == 1 && kernel_y == 1 && ic_g_r4 % 8 == 0 && stride_x == 1 && stride_y == 1 &&
                     pad_x == 0 && pad_y == 0 && dims_input[2] * dims_input[3] % 4 == 0;
    if (!no_im2col) {
        im_col_func_ = im2col;
        if (ic_g == 1) {
            im_col_func_ = im2col_smallc<1>;
        } else if (ic_g == 2) {
            im_col_func_ = im2col_smallc<2>;
        } else if (ic_g == 3) {
            im_col_func_ = im2col_smallc<3>;
        }
    } else {
        im_col_func_ = nullptr;
    }

    // set tile blk size, which be limit to 16KB
    // 16 * 1024 / sizeof(int8_t)
    int tile_blk = 16384 / (ic_g_r4 * kernel_x * kernel_y);
    tile_blk = ROUND_UP(tile_blk, SIMD_INT8CONV_TILE_HW);
    if (tile_blk < SIMD_INT8CONV_TILE_HW) {
        tile_blk = SIMD_INT8CONV_TILE_HW;
    }
    if (tile_blk > 512) {
        tile_blk = 512;
    }
    tile_blk_ = tile_blk;

    RETURN_ON_NEQ(allocateBufferWeight(inputs, outputs), TNN_OK);

    return TNN_OK;
}

void GemmInt8(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias,
              const float* scale, int hw_tile, int src_depth_d8, int src_w_step, int dst_depth, int relu,
              const int8_t* add_input, const float* add_scale, const int8_t* relu6_max, x86_isa_t arch) {
    const int src_depth_d16 = UP_DIV(src_depth_d8, 2);

    auto gemm_kernel = X86SSEGemmInt8Unit4x4;
#ifdef __AVX2__
    if (arch == avx2) {
        gemm_kernel = X86AVXGemmInt8Unit4x4;
    }
#endif

    for (int j = 0; j < dst_depth; j += 4) {
        int hw = 0;
        for (; hw + SIMD_INT8CONV_TILE_HW - 1 < hw_tile; hw += SIMD_INT8CONV_TILE_HW) {
            auto src_hw = src + hw * src_w_step;
            auto dst_hw = dst + hw * dst_depth;
            auto add_input_hw = add_input ? add_input + hw * dst_depth : nullptr;
            gemm_kernel(src_hw, weight, dst_hw, src_w_step, dst_depth, src_depth_d8,
                        scale + j, bias + j, relu, add_input_hw, add_scale, relu6_max);
        }
        if (hw < hw_tile) {
            auto src_hw = src + hw * src_w_step;
            auto dst_hw = dst + hw * dst_depth;
            auto add_input_hw = add_input ? add_input + hw * dst_depth : nullptr;
            int real_hw_tile = hw_tile - hw;

            int8_t outptr_tmp[16] = {0};
            int8_t add_input_tmp[16] = {0};
            int8_t * add_input_ptr_tmp = nullptr;

            if (add_input) {
                add_input_ptr_tmp = add_input_tmp;
                for (int i = 0; i < real_hw_tile; i++) {
                    memcpy(add_input_ptr_tmp + i * 4, add_input_hw + i * dst_depth, 4 * sizeof(int8_t));
                }
            }
            gemm_kernel(src_hw, weight, outptr_tmp, src_w_step, 4, src_depth_d8,
                        scale + j, bias + j, relu, add_input_ptr_tmp, add_scale, relu6_max);

            for (int i = 0; i < real_hw_tile; i++) {
                memcpy(dst_hw + i * dst_depth, outptr_tmp + i * 4, 4 * sizeof(int8_t));
            }
        }

        dst += 4;
        weight += 4 * src_depth_d16 * 16;
        if (add_input) {
            add_input += 4;
            add_scale += 4;
        }
        if (relu6_max) {
            relu6_max += 4;
        }
    }
}

static inline void memcpy_2d(int8_t *dst, int8_t *src, int height, int width, int dst_stride, int src_stride) {
    for (int h = 0; h < height; h++) {
        memcpy(dst + h * dst_stride, src + h * src_stride, width);
    }
}

Status X86ConvInt8LayerCommon::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
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
    auto ic_g        = dims_input[1] / conv_param->group;
    auto ic_g_r4     = ROUND_UP(ic_g, 4);
    auto oc_g        = dims_output[1] / conv_param->group;
    auto oc_g_r4     = ROUND_UP(oc_g, 4);
    auto ic_calc     = ic_g < 4 ? ic_g : ic_g_r4;

    auto input_channel_stride  = DimsVectorUtils::Count(dims_input, 2);
    auto output_channel_stride = DimsVectorUtils::Count(dims_output, 2);
    auto input_batch_stride    = input_channel_stride * ic_r4;
    auto output_batch_stride   = output_channel_stride * oc_r4;
    auto kernel_group_stride   = oc_g_r4 * ROUND_UP(ic_g_r4 * conv_param->kernels[0] * conv_param->kernels[1], 16);

    int8_t *input_data     = handle_ptr<int8_t *>(input->GetHandle());
    int8_t *output_data    = handle_ptr<int8_t *>(output->GetHandle());
    int8_t *add_input_data = add_input ? handle_ptr<int8_t *>(add_input->GetHandle()) : nullptr;

    float *scale_ptr   = buffer_scale_.force_to<float *>();
    int32_t *bias_ptr  = buffer_bias_.force_to<int32_t *>();
    int8_t *weight_ptr = buffer_weight_.force_to<int8_t *>();

    const int crs_div8   = UP_DIV(ic_calc * conv_param->kernels[1] * conv_param->kernels[0], 8);
    int tile_count = UP_DIV(dims_output[2] * dims_output[3], tile_blk_);

    // for multi-threads, adjust tile_blk to make more threads parallel
    int max_num_threads = OMP_MAX_THREADS_NUM_;
    if (max_num_threads > 1) {
        while (tile_count < max_num_threads && tile_blk_ > SIMD_INT8CONV_TILE_HW) {
            tile_blk_ = ROUND_UP(tile_blk_ / 2, SIMD_INT8CONV_TILE_HW);
            tile_count = UP_DIV(dims_output[2] * dims_output[3], tile_blk_);
        }
    }

    size_t workspace_size = 0;
    size_t src_group_buf_size = 0;
    size_t dst_group_buf_size = 0;
    if (conv_param->group > 1) {
        src_group_buf_size = ic_g_r4 * input_channel_stride;
        dst_group_buf_size = oc_g_r4 * output_channel_stride;
        dims_input[1] = ic_g;
    }
    size_t im2col_buf_size = ROUND_UP(ic_g_r4 * conv_param->kernels[0] * conv_param->kernels[1], 16) *
                             tile_blk_ * max_num_threads + SIMD_KERNEL_EXTRA_LOAD;
    workspace_size = src_group_buf_size + dst_group_buf_size + im2col_buf_size;
    int8_t *workspace = reinterpret_cast<int8_t *>(context_->GetSharedWorkSpace(workspace_size));
    // im2col will memset 0 by itself
    memset(workspace, 0, src_group_buf_size + dst_group_buf_size);
    auto im2col_buf_ptr = workspace + src_group_buf_size + dst_group_buf_size;

    for (int n = 0; n < batch; ++n) {
        auto input_batch     = input_data + n * input_batch_stride;
        auto output_batch    = output_data + n * output_batch_stride;
        auto add_input_batch = add_input_data ? add_input_data + n * output_batch_stride : nullptr;

        auto input_group  = input_batch;
        auto output_group = output_batch;
        if (conv_param->group > 1) {
            input_group  = workspace;
            output_group = workspace + src_group_buf_size;
        }

        for (int g = 0; g < conv_param->group; g++) {
            if (conv_param->group > 1) {
                auto input_ptr = input_batch + g * ic_g;
                memcpy_2d(input_group, input_ptr, input_channel_stride, ic_g, ic_g_r4, ic_r4);
            }
            auto bias_g      = bias_ptr + g * oc_g;
            auto scale_g     = scale_ptr + g * oc_g;
            auto relu6_max_g = relu6_max_.force_to<int8_t *>() + g * oc_g;
            auto weight_g    = weight_ptr + g * kernel_group_stride;

            OMP_PARALLEL_FOR_GUIDED_
            for (int t_idx = 0; t_idx < tile_count; t_idx++) {
                int thread_id          = OMP_TID_;
                int8_t *input_kernel   = nullptr;
                const int hw_start     = t_idx * tile_blk_;
                const int real_hw_tile = MIN(output_channel_stride - hw_start, tile_blk_);
                const int input_count  = crs_div8 * tile_blk_ * 8;

                // im2col
                if (im_col_func_) {
                    input_kernel = im2col_buf_ptr + input_count * thread_id;
                    im_col_func_(input_kernel, input_group, conv_param,
                                 hw_start, real_hw_tile, crs_div8, dims_input, dims_output);
                } else {
                    input_kernel = input_group + hw_start * ic_calc;
                }
                auto output_kernel    = output_group + hw_start * oc_g_r4;
                // add_input not support group conv
                auto add_input_kernel = add_input_batch ? add_input_batch + hw_start * oc_g_r4 : nullptr;

                GemmInt8(output_kernel, input_kernel, weight_g, bias_g, scale_g,
                         real_hw_tile, crs_div8, crs_div8 * 8, oc_g_r4, relu_,
                         add_input_kernel, buffer_add_scale_.force_to<float *>(),
                         relu6_max_g, arch_);
            }

            if (conv_param->group > 1) {
                auto output_ptr = output_batch + g * oc_g;
                memcpy_2d(output_ptr, output_group, output_channel_stride, oc_g, oc_r4, oc_g_r4);

                if (oc_r4 != dims_output[1] && g == conv_param->group - 1) {
                    auto output_ptr_corner_case = output_batch + dims_output[1];
                    for (int h = 0; h < output_channel_stride; h++) {
                        memset(output_ptr_corner_case + h * oc_r4, 0, (oc_r4 - dims_output[1]));
                    }
                }
            }
        }
    }

    return TNN_OK;
}

}  // namespace TNN_NS
