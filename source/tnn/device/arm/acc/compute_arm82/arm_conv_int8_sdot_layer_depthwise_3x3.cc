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

#include "tnn/device/arm/acc/convolution/arm_conv_int8_sdot_layer_depthwise_3x3.h"
#include "tnn/device/arm/acc/compute_arm82/compute_sdot_int8.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/cpu_utils.h"

#include "tnn/utils/omp_utils.h"

#ifdef TNN_ARM82_USE_NEON
namespace TNN_NS {

Status ArmConvInt8SdotLayerDepthwise3x3::allocateBufferWeight(const std::vector<Blob *> &inputs,
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
        int oc_r4  = ROUND_UP(oc, 4);

        int weight_count     = oc_r4 * 12;
        int weight_byte_size = weight_count * DataTypeUtils::GetBytesSize(conv_res->filter_handle.GetDataType());
        RawBuffer temp_buffer(weight_byte_size + NEON_KERNEL_EXTRA_LOAD);

        auto weight_src = conv_res->filter_handle.force_to<int8_t *>();
        // temp_buffer has been memset to 0
        auto weight_dst = temp_buffer.force_to<int8_t *>();
        
        PackSDOTDW3X3INT8Weight(weight_src, weight_dst, oc);

        buffer_weight_ = temp_buffer;
    }

    return TNN_OK;
}

bool ArmConvInt8SdotLayerDepthwise3x3::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                         const std::vector<Blob *> &outputs) {
    if (!param) {
        return false;
    }

    auto dims_input          = inputs[0]->GetBlobDesc().dims;
    auto dims_output         = outputs[0]->GetBlobDesc().dims;
    const int input_channel  = dims_input[1];
    const int output_channel = dims_output[1];

    bool support_dot = CpuUtils::CpuSupportInt8Dot();
    // only support convdw3x3 stride1/2 pad1 dialation1
    return (param->group == input_channel && param->group == output_channel) &&
           (param->kernels[0] == param->kernels[1] && (param->kernels[0] == 3)) &&
           (param->dialations[0] == 1 && param->dialations[1] == 1) &&
           ((param->strides[0] == 1 && param->strides[1] == 1) ||
            (param->strides[0] == 2 && param->strides[1] == 2)) &&
           (param->pads[0] == param->pads[1] && param->pads[0] == param->pads[2] &&
            param->pads[0] == param->pads[3] && param->pads[0] == 1) &&
           (param->fusion_type == FusionType_None) && support_dot;
}

ArmConvInt8SdotLayerDepthwise3x3::~ArmConvInt8SdotLayerDepthwise3x3() {}

static inline void cache_lines_slide_s1(int8_t **cache_lines, int n) {
    auto temp = cache_lines[0];
    for (int i = 0; i < n - 1; i++) {
        cache_lines[i] = cache_lines[i + 1];
    }
    cache_lines[n - 1] = temp;
}

static inline void cache_lines_slide_s2(int8_t **cache_lines, int n) {
    auto temp0 = cache_lines[0];
    auto temp1 = cache_lines[1];
    for (int i = 0; i < n - 2; i++) {
        cache_lines[i] = cache_lines[i + 2];
    }
    cache_lines[n - 2] = temp0;
    cache_lines[n - 1] = temp1;
}

static void DepthwiseI8K3S1Sdot(int8_t* dst, int8_t** src, const int8_t* weight, const int32_t* bias_z, long width,
                              long dst_depth, const float* scale_z, const int8_t* relu6_max, int activation_type) {
    OMP_PARALLEL_FOR_GUIDED_
    for (long dc = 0; dc < dst_depth - 7; dc += 8) {
        ConvDw3x3Int8SdotSlideW(dst + dc, src, weight + dc * 12, bias_z + dc, scale_z + dc, dc, dst_depth, width);
    }
    long dc = dst_depth / 8 * 8;
    if (dc < dst_depth) {
        ConvDw3x3Int8SdotSlideWLeftC4(dst + dc, src, weight + dc * 12, bias_z + dc, scale_z + dc, dc, dst_depth, width);
    }

    if (activation_type == ActivationType_ReLU) {
        ReluInt8(dst, dst, dst_depth * width);
    } else if (activation_type == ActivationType_ReLU6) {
        Relu6Int8(dst, dst, relu6_max, width, dst_depth);
    }
}

static void DepthwiseI8K3S2Sdot(int8_t* dst, int8_t** src, const int8_t* weight, const int32_t* bias_z, long width,
                              long dst_depth, const float* scale_z, const int8_t* relu6_max, int activation_type) {
    OMP_PARALLEL_FOR_GUIDED_
    for (long dc = 0; dc < dst_depth - 7; dc += 8) {
        ConvDw3x3S2Int8SdotSlideW(dst + dc, src, weight + dc * 12, bias_z + dc, scale_z + dc, dc, dst_depth, width);
    }
    long dc = dst_depth / 8 * 8;
    if (dc < dst_depth) {
        ConvDw3x3S2Int8SdotSlideWLeftC4(dst + dc, src, weight + dc * 12, bias_z + dc, scale_z + dc, dc, dst_depth, width);
    }

    if (activation_type == ActivationType_ReLU) {
        ReluInt8(dst, dst, dst_depth * width);
    } else if (activation_type == ActivationType_ReLU6) {
        Relu6Int8(dst, dst, relu6_max, width, dst_depth);
    }
}

static void DepthwiseK3S1SlideW(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias,
                                const float* scale, const int8_t* relu6_max, int8_t* work_space,
                                int8_t** cache_line, ArmKernelParam* k_param, int activation_type,
                                int batch, int workspace_w_stride) {
    auto src_h_stride = k_param->iw * k_param->ic_r4;
    auto dst_h_stride = k_param->ow * k_param->oc_r4;
    int pad_l = 1, pad_t = 1, pad_b = 1;
    int kernel_h = 3;

    for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
        auto src_ptr = src + batch_idx * k_param->ih * src_h_stride;
        auto dst_ptr = dst + batch_idx * k_param->oh * dst_h_stride;

        for (int h = 0; h < kernel_h; h++) {
            cache_line[h] = work_space + h * workspace_w_stride;
        }

        // memset top pad_t if batch > 0, batch = 0 already memset 0
        if (batch_idx > 0) {
            for (int h = 0; h < pad_t; h++) {
                memset(cache_line[h] + pad_l * k_param->oc_r4, 0, sizeof(int8_t) * src_h_stride);
            }
        }

        for (int h = pad_t; h < kernel_h - 1; h++) {
            auto cache_line_h_ptr = cache_line[h] + pad_l * k_param->oc_r4;
            memcpy(cache_line[h] + pad_l * k_param->oc_r4, src_ptr, sizeof(int8_t) * src_h_stride);
            src_ptr += src_h_stride;
        }

        int cache_line_idx = kernel_h - 1;
        for (int h = 0; h < k_param->oh - pad_b; h++) {
            memcpy(cache_line[cache_line_idx] + pad_l * k_param->oc_r4, src_ptr, sizeof(int8_t) * src_h_stride);

            DepthwiseI8K3S1Sdot(dst_ptr, cache_line, weight, bias, k_param->ow, k_param->oc_r4,
                              scale, relu6_max, activation_type);

            src_ptr += src_h_stride;
            dst_ptr += dst_h_stride;
            cache_lines_slide_s1(cache_line, kernel_h);
        }

        for (int h = pad_b; h > 0; h--) {
            memset(cache_line[cache_line_idx] + pad_l * k_param->oc_r4, 0, sizeof(int8_t) * src_h_stride);

            DepthwiseI8K3S1Sdot(dst_ptr, cache_line, weight, bias, k_param->ow, k_param->oc_r4,
                              scale, relu6_max, activation_type);

            dst_ptr += dst_h_stride;
            cache_lines_slide_s1(cache_line, kernel_h);
        }
    }
}

static void DepthwiseK3S2SlideW(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias,
                                const float* scale, const int8_t* relu6_max, int8_t* work_space,
                                int8_t** cache_line, ArmKernelParam* k_param, int activation_type,
                                int batch, int workspace_w_stride) {
    auto src_h_stride = k_param->iw * k_param->ic_r4;
    auto dst_h_stride = k_param->ow * k_param->oc_r4;
    int pad_l = 1, pad_t = 1, pad_b = 1;
    int kernel_h = 3;

    for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
        auto src_ptr = src + batch_idx * k_param->ih * src_h_stride;
        auto dst_ptr = dst + batch_idx * k_param->oh * dst_h_stride;

        for (int h = 0; h < kernel_h; h++) {
            cache_line[h] = work_space + h * workspace_w_stride;
        }

        // memset top pad_t if batch > 0, batch = 0 already memset 0
        if (batch_idx > 0) {
            for (int h = 0; h < pad_t; h++) {
                memset(cache_line[h] + pad_l * k_param->oc_r4, 0, sizeof(int8_t) * src_h_stride);
            }
        }

        int cache_line_idx_0 = kernel_h - 2;
        int cache_line_idx_1 = kernel_h - 1;
        for (int h = 0; h < k_param->oh - pad_b; h++) {
            memcpy(cache_line[cache_line_idx_0] + pad_l * k_param->oc_r4, src_ptr, sizeof(int8_t) * src_h_stride);
            memcpy(cache_line[cache_line_idx_1] + pad_l * k_param->oc_r4, src_ptr + src_h_stride, sizeof(int8_t) * src_h_stride);

            DepthwiseI8K3S2Sdot(dst_ptr, cache_line, weight, bias, k_param->ow, k_param->oc_r4,
                              scale, relu6_max, activation_type);

            src_ptr += 2 * src_h_stride;
            dst_ptr += dst_h_stride;
            cache_lines_slide_s2(cache_line, kernel_h);
        }

        // corner case oh - 1
        int h = k_param->oh - pad_b;
        int ih_end = h * 2 - pad_l + 3 - 1;
        if (ih_end > k_param->ih - 1) {
            memcpy(cache_line[cache_line_idx_0] + pad_l * k_param->oc_r4, src_ptr, sizeof(int8_t) * src_h_stride);
            memset(cache_line[cache_line_idx_1] + pad_l * k_param->oc_r4, 0, sizeof(int8_t) * src_h_stride);

            DepthwiseI8K3S2Sdot(dst_ptr, cache_line, weight, bias, k_param->ow, k_param->oc_r4,
                              scale, relu6_max, activation_type);
        } else {
            memcpy(cache_line[cache_line_idx_0] + pad_l * k_param->oc_r4, src_ptr, sizeof(int8_t) * src_h_stride);
            memcpy(cache_line[cache_line_idx_1] + pad_l * k_param->oc_r4, src_ptr + src_h_stride, sizeof(int8_t) * src_h_stride);

            DepthwiseI8K3S2Sdot(dst_ptr, cache_line, weight, bias, k_param->ow, k_param->oc_r4,
                              scale, relu6_max, activation_type);
        }
    }
}

Status ArmConvInt8SdotLayerDepthwise3x3::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);

    auto input       = inputs[0];
    auto output      = outputs[0];
    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;

    DataType data_type = output->GetBlobDesc().data_type;
    int data_byte_size = DataTypeUtils::GetBytesSize(data_type);

    const int batch    = dims_output[0];
    int pad_l          = conv_param->pads[0];
    int pad_r          = conv_param->pads[1];
    int pad_t          = conv_param->pads[2];
    int pad_b          = conv_param->pads[3];

    auto *src_origin     = reinterpret_cast<const int8_t *>(GetBlobHandlePtr(input->GetHandle()));
    auto *dst_origin     = reinterpret_cast<int8_t *>(GetBlobHandlePtr(output->GetHandle()));
    auto src_h_stride    = k_param_->iw * k_param_->ic_r4;
    auto dst_h_stride    = k_param_->ow * k_param_->oc_r4;
    auto workspace_w_pad = k_param_->iw + pad_l + pad_r;
    int extra_load = 0;
    // 1 for extra preload in f3s2 asm kernel
    if (conv_param->strides[0] == 2) extra_load = 1;
    auto workspace_w_stride   = (workspace_w_pad + extra_load) * k_param_->oc_r4;
    auto workspace_size = conv_param->kernels[1] * workspace_w_stride * data_byte_size;

    if (pad_t > conv_param->kernels[1]) {
        LOGE("ERROR: ConvDw pad_t must small than kernel_h\n");
        return Status(TNNERR_LAYER_ERR, "ERROR: ConvDw pad_t must small than kernel_h");
    }

    auto work_space = reinterpret_cast<int8_t *>(context_->GetSharedWorkSpace(workspace_size));
    int8_t** cache_line = (int8_t**)malloc(conv_param->kernels[1] * sizeof(int8_t*));

    float *scale_ptr      = buffer_scale_.force_to<float *>();
    int32_t *bias_ptr     = buffer_bias_.force_to<int32_t *>();
    int8_t *weight_ptr    = buffer_weight_.force_to<int8_t *>();
    int8_t *relu6_max_ptr = relu6_max_.force_to<int8_t *>();

    // data in workspace are dirty, must be clear first for left and right padding area
    memset(work_space, 0, workspace_size);

    if (conv_param->strides[0] == 1) {
        DepthwiseK3S1SlideW(dst_origin, src_origin, weight_ptr, bias_ptr, scale_ptr, relu6_max_ptr,
                            work_space, cache_line, k_param_.get(), conv_param->activation_type,
                            batch, workspace_w_stride);
    } else if (conv_param->strides[0] == 2) {
        DepthwiseK3S2SlideW(dst_origin, src_origin, weight_ptr, bias_ptr, scale_ptr, relu6_max_ptr,
                            work_space, cache_line, k_param_.get(), conv_param->activation_type,
                            batch, workspace_w_stride);
    }

    free(cache_line);
    return TNN_OK;
}

}  // namespace TNN_NS
#endif
#endif
