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
#include "tnn/device/arm/acc/convolution/arm_conv_fp16_layer_depthwise_s1.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"

#include "tnn/utils/omp_utils.h"
#include "tnn/device/arm/acc/Half8.h"

#define MAX_CACHE_LINE_NUM 7

namespace TNN_NS {

template <typename T>
static inline void cache_lines_slide(T **cache_lines, int n) {
    auto temp = cache_lines[0];
    for (int i = 0; i < n - 1; i++) {
        cache_lines[i] = cache_lines[i + 1];
    }
    cache_lines[n - 1] = temp;
}

bool ArmConvFp16LayerDepthwiseS1::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                         const std::vector<Blob *> &outputs) {
    if (!param) {
        return false;
    }

    auto dims_input          = inputs[0]->GetBlobDesc().dims;
    auto dims_output         = outputs[0]->GetBlobDesc().dims;
    const int input_channel  = dims_input[1];
    const int output_channel = dims_output[1];

    // only support convdw3x3
    return param->group == input_channel && param->group == output_channel &&
           (param->kernels[0] == param->kernels[1] && (param->kernels[0] == 3)) &&
           param->dialations[0] == 1 && param->dialations[1] == 1 && param->strides[0] == 1 && param->strides[1] == 1;
}

ArmConvFp16LayerDepthwiseS1::~ArmConvFp16LayerDepthwiseS1() {}

Status ArmConvFp16LayerDepthwiseS1::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ArmConvFp16LayerDepthwise::Reshape(inputs, outputs);
    auto in_data_type = inputs[0]->GetBlobDesc().data_type;

    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    if (conv_param) {
        if (in_data_type == DATA_TYPE_HALF) {
            if (conv_param->kernels[1] == 3) {
                SlideFunc_ = ConvDw3x3Fp16SlideW;
            } else {
                return TNNERR_LAYER_ERR;
            }
        } else {
            return TNNERR_LAYER_ERR;
        }
    }
    return TNN_OK;
}

Status ArmConvFp16LayerDepthwiseS1::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);

    auto input       = inputs[0];
    auto output      = outputs[0];
    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;

    DataType data_type = output->GetBlobDesc().data_type;
    int data_byte_size = DataTypeUtils::GetBytesSize(data_type);

    const int batch    = dims_output[0];
    int dst_z_step     = k_param_->ow * k_param_->oh;
    int src_z_step     = k_param_->iw * k_param_->ih;
    int pad_l          = conv_param->pads[0];
    int pad_r          = conv_param->pads[1];
    int pad_t          = conv_param->pads[2];
    int pad_b          = conv_param->pads[3];
    int weight_z_step  = conv_param->kernels[0] * conv_param->kernels[1];

    const fp16_t *src_origin = reinterpret_cast<const fp16_t *>(GetBlobHandlePtr(input->GetHandle()));
    fp16_t *dst_origin       = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(output->GetHandle()));
    int max_num_threads      = OMP_MAX_THREADS_NUM_;
    int workspace_per_thread = conv_param->kernels[1] * (k_param_->iw + pad_l + pad_r) * 8 * data_byte_size;

    if (!SlideFunc_) {
        LOGE("Error: ConvDw slide func is nil\n");
        return Status(TNNERR_LAYER_ERR, "Error: ConvDw slide func is nil");
    }

    if (pad_t > conv_param->kernels[1]) {
        LOGE("ERROR: ConvDw pad_t must small than kernel_h\n");
        return Status(TNNERR_LAYER_ERR, "ERROR: ConvDw pad_t must small than kernel_h");
    }

    fp16_t *work_space = reinterpret_cast<fp16_t *>(context_->GetSharedWorkSpace(max_num_threads * workspace_per_thread));

    /*
    [ATTENTION]
    data in workspace are dirty, must be clear first
    */
    memset(work_space, 0, max_num_threads * workspace_per_thread);
    for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
        auto src_ptr = src_origin + batch_idx * k_param_->iw * k_param_->ih * k_param_->ic_r8;
        auto dst_ptr = dst_origin + batch_idx * k_param_->ow * k_param_->oh * k_param_->oc_r8;

        OMP_PARALLEL_FOR_
        for (int dz = 0; dz < k_param_->oc_r8; dz += 8) {
            auto *dst_z                       = dst_ptr + dst_z_step * dz;
            auto *src_z                       = src_ptr + src_z_step * dz;
            const auto *weight_dz             = reinterpret_cast<fp16_t *>(k_param_->fil_ptr) + dz * weight_z_step;
            int thread_id                     = OMP_TID_;
            auto thread_work_space            = work_space + thread_id * workspace_per_thread / data_byte_size;
            fp16_t *cache_line[MAX_CACHE_LINE_NUM] = {nullptr};
            for (int i = 0; i < conv_param->kernels[1]; i++) {
                cache_line[i] = thread_work_space + i * (k_param_->iw + pad_l + pad_r) * 8;
            }

            auto src_y = src_z;
            auto dst_y = dst_z;
            // memset pat_t lines
            for (int ky = 0; ky < pad_t; ky++) {
                memset(cache_line[ky] + pad_l * 8, 0, k_param_->iw * 8 * data_byte_size);
            }
            // load mid lines
            for (int ky = pad_t; ky < conv_param->kernels[1] - 1; ky++) {
                memcpy(cache_line[ky] + pad_l * 8, src_y, k_param_->iw * 8 * data_byte_size);
                src_y += k_param_->iw * 8;
            }
            for (int dy = 0; dy < k_param_->oh - pad_b; dy++) {
                // load only one line every loop
                memcpy(cache_line[conv_param->kernels[1] - 1] + pad_l * 8, src_y, k_param_->iw * 8 * data_byte_size);
                // kernel func
                SlideFunc_(dst_y, (void **)cache_line, weight_dz, k_param_->ow);

                src_y += k_param_->iw * 8;
                dst_y += k_param_->ow * 8;
                cache_lines_slide(cache_line, conv_param->kernels[1]);
            }
            // memset pad_b lines
            for (int ky = pad_b; ky > 0; ky--) {
                memset(cache_line[conv_param->kernels[1] - 1] + pad_l * 8, 0, k_param_->iw * 8 * data_byte_size);
                // kernel func
                SlideFunc_(dst_y, (void **)cache_line, weight_dz, k_param_->ow);

                dst_y += k_param_->ow * 8;
                cache_lines_slide(cache_line, conv_param->kernels[1]);
            }
        }
    }

    PostExec<fp16_t>(outputs);

    return TNN_OK;
}

}  // namespace TNN_NS
#endif
