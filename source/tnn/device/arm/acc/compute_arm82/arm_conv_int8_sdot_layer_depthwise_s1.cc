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

#include "tnn/device/arm/acc/convolution/arm_conv_int8_sdot_layer_depthwise_s1.h"
#include "tnn/device/arm/acc/compute_arm82/compute_sdot_int8.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/cpu_utils.h"

#include "tnn/utils/omp_utils.h"

#ifdef TNN_ARM82_A64

namespace TNN_NS {

Status ArmConvInt8SdotLayerDepthwiseS1::allocateBufferWeight(const std::vector<Blob *> &inputs,
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

bool ArmConvInt8SdotLayerDepthwiseS1::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                         const std::vector<Blob *> &outputs) {
    if (!param) {
        return false;
    }

    auto dims_input          = inputs[0]->GetBlobDesc().dims;
    auto dims_output         = outputs[0]->GetBlobDesc().dims;
    const int input_channel  = dims_input[1];
    const int output_channel = dims_output[1];

    bool support_dot = CpuUtils::CpuSupportInt8Dot();
    // only support convdw3x3 stride1 pad1 dialation1
    return (param->group == input_channel && param->group == output_channel) &&
           (param->kernels[0] == param->kernels[1] && (param->kernels[0] == 3)) &&
           (param->dialations[0] == 1 && param->dialations[1] == 1) &&
           (param->strides[0] == 1 && param->strides[1] == 1) &&
           (param->pads[0] == param->pads[1] && param->pads[0] == param->pads[2] &&
            param->pads[0] == param->pads[3] && param->pads[0] == 1) &&
           (param->fusion_type == FusionType_None) && support_dot;
}

ArmConvInt8SdotLayerDepthwiseS1::~ArmConvInt8SdotLayerDepthwiseS1() {}

static inline void cache_lines_slide(int8_t **cache_lines, int n) {
    auto temp = cache_lines[0];
    for (int i = 0; i < n - 1; i++) {
        cache_lines[i] = cache_lines[i + 1];
    }
    cache_lines[n - 1] = temp;
}

static void DepthwiseI8K3Sdot(int8_t* dst, int8_t** src, const int8_t* weight, const int32_t* bias_z, long width,
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

Status ArmConvInt8SdotLayerDepthwiseS1::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
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
    int weight_z_step  = conv_param->kernels[0] * conv_param->kernels[1];

    auto *src_origin     = reinterpret_cast<const int8_t *>(GetBlobHandlePtr(input->GetHandle()));
    auto *dst_origin     = reinterpret_cast<int8_t *>(GetBlobHandlePtr(output->GetHandle()));
    int max_num_threads  = OMP_MAX_THREADS_NUM_;
    auto src_h_stride    = k_param_->iw * k_param_->ic_r4;
    auto dst_h_stride    = k_param_->ow * k_param_->oc_r4;
    auto workspace_w_pad = k_param_->iw + pad_l + pad_r;
    // 2 for extra preload in asm kernel
    auto workspace_w_stride   = (workspace_w_pad + 2) * k_param_->oc_r4;
    auto workspace_per_thread = conv_param->kernels[1] * workspace_w_stride * data_byte_size;

    if (pad_t > conv_param->kernels[1]) {
        LOGE("ERROR: ConvDw pad_t must small than kernel_h\n");
        return Status(TNNERR_LAYER_ERR, "ERROR: ConvDw pad_t must small than kernel_h");
    }

    auto work_space = reinterpret_cast<int8_t *>(context_->GetSharedWorkSpace(max_num_threads * workspace_per_thread));
    int8_t** cache_line = (int8_t**)malloc(conv_param->kernels[1] * sizeof(int8_t*));

    float *scale_ptr      = buffer_scale_.force_to<float *>();
    int32_t *bias_ptr     = buffer_bias_.force_to<int32_t *>();
    int8_t *weight_ptr    = buffer_weight_.force_to<int8_t *>();
    int8_t *relu6_max_ptr = relu6_max_.force_to<int8_t *>();

    // data in workspace are dirty, must be clear first for left and right padding area
    memset(work_space, 0, max_num_threads * workspace_per_thread);
    for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
        auto src_ptr = src_origin + batch_idx * k_param_->ih * src_h_stride;
        auto dst_ptr = dst_origin + batch_idx * k_param_->oh * dst_h_stride;

        for (int h = 0; h < conv_param->kernels[1]; h++) {
            cache_line[h] = work_space + h * workspace_w_stride;
        }

        // memset top pad_t if batch > 0, batch = 0 already memset 0
        if (batch_idx > 0) {
            for (int h = 0; h < pad_t; h++) {
                memset(cache_line[h] + pad_l * k_param_->oc_r4, 0, sizeof(int8_t) * src_h_stride);
            }
        }

        for (int h = pad_t; h < conv_param->kernels[1] - 1; h++) {
            auto cache_line_h_ptr = cache_line[h] + pad_l * k_param_->oc_r4;
            memcpy(cache_line[h] + pad_l * k_param_->oc_r4, src_ptr, sizeof(int8_t) * src_h_stride);
            src_ptr += src_h_stride;
        }

        int cache_line_idx = conv_param->kernels[1] - 1;
        for (int h = 0; h < k_param_->oh - pad_b; h++) {
            memcpy(cache_line[cache_line_idx] + pad_l * k_param_->oc_r4, src_ptr, sizeof(int8_t) * src_h_stride);

            DepthwiseI8K3Sdot(dst_ptr, cache_line, weight_ptr, bias_ptr, k_param_->ow, k_param_->oc_r4,
                              scale_ptr, relu6_max_ptr, conv_param->activation_type);

            src_ptr += src_h_stride;
            dst_ptr += dst_h_stride;
            cache_lines_slide(cache_line, conv_param->kernels[1]);
        }

        for (int h = pad_b; h > 0; h--) {
            memset(cache_line[cache_line_idx] + pad_l * k_param_->oc_r4, 0, sizeof(int8_t) * src_h_stride);

            DepthwiseI8K3Sdot(dst_ptr, cache_line, weight_ptr, bias_ptr, k_param_->ow, k_param_->oc_r4,
                              scale_ptr, relu6_max_ptr, conv_param->activation_type);

            dst_ptr += dst_h_stride;
            cache_lines_slide(cache_line, conv_param->kernels[1]);
        }
    }

    return TNN_OK;
}

}  // namespace TNN_NS
#endif
#endif
