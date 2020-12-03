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

#include "tnn/device/arm/acc/convolution/arm_conv_int8_layer_common.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {
/*
ArmConvInt8LayerCommon as the last conv int8 solution
*/
bool ArmConvInt8LayerCommon::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs) {
    if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_INT8) {
        return true;
    }
    return false;
}

ArmConvInt8LayerCommon::~ArmConvInt8LayerCommon() {}

Status ArmConvInt8LayerCommon::allocateBufferWeight(const std::vector<Blob *> &inputs,
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
            return Status(TNNERR_PARAM_ERR, "INT8 CONV GROUD > 1 NOT SUPPORT");
        }
        const int oc      = output_channel;
        const int ic      = inputs[0]->GetBlobDesc().dims[1];
        const int oc_4    = UP_DIV(oc, 4);
        const int ic_4    = UP_DIV(ic, 4);
        const int icrs_16 = UP_DIV(ic_4 * kw * kh, 4);

        int weight_count   = group * oc_4 * icrs_16 * 64;
        int data_byte_size = weight_count * DataTypeUtils::GetBytesSize(conv_res->filter_handle.GetDataType());
        RawBuffer temp_buffer(data_byte_size + NEON_KERNEL_EXTRA_LOAD);

        // from [o][i][h][w]
        // to armv8: [o/4][h][w][i/16][o4][i16]
        // to armv7: [o/4][h][w][i/8][o2][i2][o2][i4]
        PackINT8Weight(conv_res->filter_handle.force_to<int8_t *>(), temp_buffer.force_to<int8_t *>(), group, ic,
                       output_channel, conv_param->kernels[1], conv_param->kernels[0]);

        buffer_weight_ = temp_buffer;
    }
    return TNN_OK;
}

Status ArmConvInt8LayerCommon::allocateBufferBias(const std::vector<Blob *> &inputs,
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

Status ArmConvInt8LayerCommon::allocateBufferScale(const std::vector<Blob *> &inputs,
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

Status ArmConvInt8LayerCommon::allocateBufferAddScale(const std::vector<Blob *> &inputs,
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

Status ArmConvInt8LayerCommon::allocateBufferParam(const std::vector<Blob *> &inputs,
                                                   const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);

    auto dims_input          = inputs[0]->GetBlobDesc().dims;
    auto dims_output         = outputs[0]->GetBlobDesc().dims;
    const int input_channel  = dims_input[1];
    const int output_channel = dims_output[1];

    int max_num_threads = OMP_CORES_;
    // alloc img2col and gemm work buffer
    if (!buffer_im2col_.GetBytesSize() || !buffer_gemm_work_space_.GetBytesSize()) {
        const int c_round4 = ROUND_UP(inputs[0]->GetBlobDesc().dims[1], 4);
        const int buffer_size =
            (ROUND_UP(c_round4 * conv_param->kernels[0] * conv_param->kernels[1], 16) * NEON_INT8CONV_TILE_HW) *
                max_num_threads +
            NEON_KERNEL_EXTRA_LOAD;

        RawBuffer temp_buffer_i2c(buffer_size);
        RawBuffer temp_buffer_ws(buffer_size);
        memset(temp_buffer_i2c.force_to<void *>(), 0, buffer_size);
        memset(temp_buffer_ws.force_to<void *>(), 0, buffer_size);
        buffer_im2col_          = temp_buffer_i2c;
        buffer_gemm_work_space_ = temp_buffer_ws;
    }
    const int oc_round4   = ROUND_UP(output_channel, 4);
    const int buffer_size = oc_round4 * NEON_INT8CONV_TILE_HW * max_num_threads;
    if (!buffer_tmpout_.GetBytesSize()) {
        RawBuffer temp_buffer(buffer_size);
        buffer_tmpout_ = temp_buffer;
    }
    if (conv_param->fusion_type != FusionType_None && !buffer_add_tmpin_.GetBytesSize()) {
        RawBuffer temp_buffer(buffer_size);
        buffer_add_tmpin_ = temp_buffer;
    }
    RETURN_ON_NEQ(allocateBufferWeight(inputs, outputs), TNN_OK);
    return TNN_OK;
}

#define DEF_IMG2COL_VAL                                                                                                \
    int x_id = (int)x_start + i;                                                                                       \
    int ox   = x_id % kparam->ow;                                                                                      \
    int oy   = x_id / kparam->ow;                                                                                      \
    int sx   = ox * param->strides[0] - param->pads[0];                                                                \
    int sy   = oy * param->strides[1] - param->pads[2];                                                                \
    int sfy  = MAX(0, (UP_DIV(-sy, param->dialations[1])));                                                            \
    int efy  = MIN(kh, UP_DIV(kparam->ih - sy, param->dialations[1]));                                                 \
    int sfx  = MAX(0, (UP_DIV(-sx, param->dialations[0])));                                                            \
    int efx  = MIN(kw, UP_DIV(kparam->iw - sx, param->dialations[0]));                                                 \
    int fyC  = efy - sfy;                                                                                              \
    int fxC  = efx - sfx;

/*
general img2col func
*/
static void im2col(int8_t *dst, const int8_t *src, const ConvLayerParam *param, size_t x_start, size_t dst_cnt,
                   int crs_div8, const ArmKernelParam *kparam) {
    const int col_buffer_size = crs_div8 * NEON_INT8CONV_TILE_HW * 8;
    const int src_w_step      = kparam->ic_r4;
    const int crs_r8          = crs_div8 * 8;
    memset(dst, 0, col_buffer_size);
    auto kh = param->kernels[1];
    auto kw = param->kernels[0];
    auto dilate_y = param->dialations[1];
    auto dilate_x = param->dialations[0];
    for (int i = 0; i < dst_cnt; ++i) {
        DEF_IMG2COL_VAL;

        auto dst_i        = dst + crs_r8 * i;
        auto input_offset = src + (sx + sfx * dilate_x + (sy + sfy * dilate_y) * kparam->iw) * src_w_step;
        auto idx_offset   = (sfy * kw + sfx) * kparam->ic_r4;
        for (int fy = 0; fy < fyC; ++fy) {
            auto dst_y = dst_i + idx_offset + fy * kw * kparam->ic_r4;
            auto src_y = input_offset + fy * kparam->iw * kparam->ic_r4 * dilate_y;
            for (int fx = 0; fx < fxC; ++fx) {
                auto dst_x = dst_y + fx * kparam->ic_r4;
                auto src_x = src_y + fx * dilate_x * kparam->ic_r4;
                memcpy(dst_x, src_x, kparam->ic_r4);
            }

        }
    }
}

/*
template img2col func when c is small(eg. 1,2,3)
*/
template <int REALC>
static void im2col_smallc(int8_t *dst, const int8_t *src, const ConvLayerParam *param, size_t x_start, size_t dst_cnt,
                          int crs_div8, const ArmKernelParam *kparam) {
    const int col_buffer_size = crs_div8 * NEON_INT8CONV_TILE_HW * 8;
    const int src_w_step      = 4;
    const int crs_r8          = crs_div8 * 8;
    memset(dst, 0, col_buffer_size);
    auto kh = param->kernels[1];
    auto kw = param->kernels[0];
    auto dilate_y = param->dialations[1];
    auto dilate_x = param->dialations[0];
    for (int i = 0; i < dst_cnt; ++i) {
        DEF_IMG2COL_VAL;

        auto dst_i        = dst + crs_r8 * i;
        auto input_offset = src + (sx + sfx * dilate_x + (sy + sfy * dilate_y) * kparam->iw) * src_w_step;
        auto idx_offset   = (sfy * kw + sfx) * REALC;
        for (int fy = 0; fy < fyC; ++fy) {
            auto dst_y = dst_i + idx_offset + fy * kw * REALC;
            auto src_y = input_offset + fy * kparam->iw * src_w_step * dilate_y;
            for (int fx = 0; fx < fxC; fx++) {
                auto dst_x = dst_y + fx * REALC;
                auto src_x = src_y + fx * src_w_step * dilate_x;
                for (int c = 0; c < REALC; c++) {
                    dst_x[c] = src_x[c];
                }
            }
        }
    }
}

Status ArmConvInt8LayerCommon::setFusionParam(const std::vector<Blob *> &inputs,
                                              const std::vector<Blob *> &outputs) {
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
    }

    return TNN_OK;
}

Status ArmConvInt8LayerCommon::Init(Context *context, LayerParam *param, LayerResource *resource,
                                    const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    RETURN_ON_NEQ(ArmLayerAcc::Init(context, param, resource, inputs, outputs), TNN_OK);
    RETURN_ON_NEQ(allocateBufferBias(inputs, outputs), TNN_OK);
    RETURN_ON_NEQ(allocateBufferScale(inputs, outputs), TNN_OK);
    RETURN_ON_NEQ(allocateBufferParam(inputs, outputs), TNN_OK);
    RETURN_ON_NEQ(setFusionParam(inputs, outputs), TNN_OK);

    // init base k_param_
    k_param_->scale   = buffer_scale_.force_to<float *>();
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

    // fast mode
    bool no_im2col = kernel_x == 1 && kernel_y == 1 && k_param_->ic_r4 % 8 == 0 && stride_x == 1 && stride_y == 1 &&
                     pad_x == 0 && pad_y == 0 && dims_input[2] * dims_input[3] % 4 == 0;
    if (!no_im2col) {
        im_col_func_ = im2col;
        if (dims_input[1] == 1)
            im_col_func_ = im2col_smallc<1>;
        else if (dims_input[1] == 2)
            im_col_func_ = im2col_smallc<2>;
        else if (dims_input[1] == 3)
            im_col_func_ = im2col_smallc<3>;
    } else {
        im_col_func_ = nullptr;
    }
    return TNN_OK;
}

Status ArmConvInt8LayerCommon::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);
    auto input  = inputs[0];
    auto output = outputs[0];
    auto add_input = (conv_param->fusion_type == FusionType_None) ? nullptr : inputs[1];

    DataType data_type = output->GetBlobDesc().data_type;
    int data_byte_size = DataTypeUtils::GetBytesSize(data_type);

    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;
    const int batch  = dims_output[0];
    auto ic          = dims_input[1];
    auto ic_calc     = ic < 4 ? ic : k_param_->ic_r4;

    int8_t *input_data  = reinterpret_cast<int8_t *>(GetBlobHandlePtr(input->GetHandle()));
    int8_t *output_data = reinterpret_cast<int8_t *>(GetBlobHandlePtr(output->GetHandle()));
    int8_t *add_input_data = add_input ? reinterpret_cast<int8_t *>(GetBlobHandlePtr(add_input->GetHandle())) : nullptr;

    const int crs_div8   = UP_DIV(ic_calc * conv_param->kernels[1] * conv_param->kernels[0], 8);
    const int tile_count = UP_DIV(k_param_->oh * k_param_->ow, NEON_INT8CONV_TILE_HW);
    for (int n = 0; n < batch; ++n) {
        const auto input_batch = input_data + n * k_param_->iw * k_param_->ih * k_param_->ic_r4;
        auto output_batch      = output_data + n * k_param_->ow * k_param_->oh * k_param_->oc_r4;
        auto add_input_batch   = add_input_data ? add_input_data + n * k_param_->ow * k_param_->oh * k_param_->oc_r4 : nullptr;

        OMP_PARALLEL_FOR_GUIDED_
        for (int t_idx = 0; t_idx < tile_count; t_idx++) {
            int thread_id          = OMP_TID_;
            int8_t *input_kernel   = nullptr;
            const int hw_start     = t_idx * NEON_INT8CONV_TILE_HW;
            const int real_hw_tile = MIN(k_param_->oh * k_param_->ow - hw_start, NEON_INT8CONV_TILE_HW);
            auto gemm_work_space   = buffer_gemm_work_space_.force_to<int8_t *>();
            // im2col
            if (im_col_func_) {
                input_kernel = buffer_im2col_.force_to<int8_t *>() + crs_div8 * NEON_INT8CONV_TILE_HW * 8 * thread_id;
                im_col_func_(input_kernel, input_batch, conv_param, hw_start, real_hw_tile, crs_div8, k_param_.get());
            } else {
                input_kernel = input_batch + hw_start * ic_calc;
            }
            auto output_kernel = output_batch + hw_start * k_param_->oc_r4;
            auto add_input_kernel = add_input_batch ? add_input_batch + hw_start * k_param_->oc_r4 : nullptr;
            // gemm int8
            if (real_hw_tile == NEON_INT8CONV_TILE_HW) {
                GemmInt8(output_kernel, input_kernel, gemm_work_space, reinterpret_cast<int8_t *>(k_param_->fil_ptr),
                         reinterpret_cast<int32_t *>(k_param_->bias), k_param_->scale, crs_div8, crs_div8 * 8,
                         k_param_->oc_r4, relu_, add_input_kernel, buffer_add_scale_.force_to<float *>());
            } else {
                int8_t *outptr_tmp =
                    buffer_tmpout_.force_to<int8_t *>() + k_param_->oc_r4 * NEON_INT8CONV_TILE_HW * thread_id;
                int8_t *add_input_ptr_tmp = nullptr;
                if (add_input_kernel) {
                    add_input_ptr_tmp = buffer_add_tmpin_.force_to<int8_t *>() + k_param_->oc_r4 * NEON_INT8CONV_TILE_HW * thread_id;
                    memcpy(add_input_ptr_tmp, add_input_kernel, real_hw_tile * k_param_->oc_r4);
                }
                GemmInt8(outptr_tmp, input_kernel, gemm_work_space, reinterpret_cast<int8_t *>(k_param_->fil_ptr),
                         reinterpret_cast<int32_t *>(k_param_->bias), k_param_->scale, crs_div8, crs_div8 * 8,
                         k_param_->oc_r4, relu_, add_input_ptr_tmp, buffer_add_scale_.force_to<float *>());
                memcpy(output_kernel, outptr_tmp, real_hw_tile * k_param_->oc_r4);
            }
        }
    }
    return TNN_OK;
}

}  // namespace TNN_NS
