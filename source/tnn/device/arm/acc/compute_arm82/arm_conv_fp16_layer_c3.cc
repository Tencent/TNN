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
#include "tnn/device/arm/acc/convolution/arm_conv_fp16_layer_c3.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/omp_utils.h"
#include "tnn/device/arm/acc/Half8.h"

namespace TNN_NS {
// usually appears on the first conv layer
bool ArmConvFp16LayerC3::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                const std::vector<Blob *> &outputs) {
    if (!param) {
        return false;
    }
    return inputs[0]->GetBlobDesc().dims[1] == 3 && param->group == 1;
}

ArmConvFp16LayerC3::~ArmConvFp16LayerC3() {}

Status ArmConvFp16LayerC3::allocateBufferWeight(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);
    ConvLayerResource *conv_res = dynamic_cast<ConvLayerResource *>(resource_);
    CHECK_PARAM_NULL(conv_res);

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;

    if (!buffer_weight_.GetBytesSize()) {
        const int ic   = dims_input[1];
        const int oc   = dims_output[1];
        const int oc_8 = UP_DIV(oc, 8);

        int kw = conv_param->kernels[0];
        int kh = conv_param->kernels[1];

        int weight_count = oc_8 * ic * kh * kw * 8;
        buffer_weight_   = RawBuffer(weight_count * DataTypeUtils::GetBytesSize(DATA_TYPE_HALF));

        if (conv_res->filter_handle.GetDataType() == DATA_TYPE_FLOAT) {
            size_t weight_nchw_count = oc * ic * kh * kw;
            RawBuffer filter_half(weight_nchw_count * DataTypeUtils::GetBytesSize(DATA_TYPE_HALF));
            Float2Half(filter_half.force_to<fp16_t *>(), conv_res->filter_handle.force_to<float *>(),
                       weight_nchw_count);
            // use int16_t to covert weights
            ConvertWeightsFromOI3HWToOHW24(filter_half.force_to<int16_t *>(), buffer_weight_.force_to<int16_t *>(),
                                           ic, oc, conv_param->kernels[1], conv_param->kernels[0]);
        } else if (conv_res->filter_handle.GetDataType() == DATA_TYPE_HALF) {
            // soft fp16 -> fp32 -> hard fp16 TBD
            ConvertWeightsFromOI3HWToOHW24(conv_res->filter_handle.force_to<int16_t *>(), buffer_weight_.force_to<int16_t *>(),
                                           ic, oc, conv_param->kernels[1], conv_param->kernels[0]);
        } else {
            LOGE("WEIGHT DATATYPE NOT SUPPORTED NOW\n");
            return Status(TNNERR_PARAM_ERR, "FP16 CONV C3 ONLY SUPPORT WEIGHT DATATYPE FLOAT AND HALF");
        }
    }
    return TNN_OK;
}

Status ArmConvFp16LayerC3::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input         = inputs[0];
    auto output        = outputs[0];
    auto dims_input    = input->GetBlobDesc().dims;
    auto dims_output   = output->GetBlobDesc().dims;
    DataType data_type = output->GetBlobDesc().data_type;
    int data_byte_size = DataTypeUtils::GetBytesSize(data_type);

    const int batch = dims_output[0];

    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    int kernel_x               = conv_param->kernels[0];
    int kernel_y               = conv_param->kernels[1];
    int dilate_y_step          = k_param_->iw * 8 * conv_param->dialations[1];
    int dilate_x_step          = 8 * conv_param->dialations[0];

    int weight_z_step = kernel_y * kernel_x * 3;

    const fp16_t *src_origin = reinterpret_cast<const fp16_t *>(GetBlobHandlePtr(input->GetHandle()));
    fp16_t *dst_origin       = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(output->GetHandle()));

    int max_num_threads = OMP_MAX_THREADS_NUM_;

    int src_xc = 1 + (k_param_->ow - 1) * conv_param->strides[0] + conv_param->dialations[0] * (kernel_x - 1);
    int workspace_per_thread = src_xc * kernel_y * k_param_->ic_r8 * data_byte_size;
    fp16_t *work_space = reinterpret_cast<fp16_t *>(context_->GetSharedWorkSpace(max_num_threads * workspace_per_thread));

    for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
        auto input_ptr  = src_origin + batch_idx * k_param_->iw * k_param_->ih * k_param_->ic_r8;
        auto output_ptr = dst_origin + batch_idx * k_param_->ow * k_param_->oh * k_param_->oc_r8;
        int src_start_x = 0 - conv_param->pads[0];
        int src_end_x   = src_start_x + src_xc >= k_param_->iw ? k_param_->iw : src_start_x + src_xc;

        int dst_offset = 0;
        if (src_start_x < 0) {
            dst_offset  = -src_start_x;
            src_start_x = 0;
        }
        int copy_count = src_end_x - src_start_x;
        auto src_x     = input_ptr + 8 * src_start_x;

        OMP_PARALLEL_FOR_
        for (int dy = 0; dy < k_param_->oh; dy++) {
            int thread_id = OMP_TID_;

            auto work_space_t = work_space + thread_id * workspace_per_thread / data_byte_size;
            memset(work_space_t, 0, workspace_per_thread);
            int src_start_y = dy * conv_param->strides[1] - conv_param->pads[2];
            int sfy         = MAX(0, (UP_DIV(-src_start_y, conv_param->dialations[1])));
            int efy         = MIN(kernel_y, UP_DIV(k_param_->ih - src_start_y, conv_param->dialations[1]));

            // copy make board
            for (int ky = sfy; ky < efy; ky++) {
                int sy     = src_start_y + ky * conv_param->dialations[1];
                auto src_y = src_x + 8 * sy * k_param_->iw;
                auto dst_y = work_space_t + (ky * src_xc + dst_offset) * 8;
                memcpy(dst_y, src_y, copy_count * 8 * data_byte_size);
            }
            for (int dz = 0; dz <= k_param_->oc_r8 - 8; dz += 8) {
                auto dst_z = output_ptr + dz * k_param_->ow * k_param_->oh + k_param_->ow * 8 * dy;
                auto weight_dz = reinterpret_cast<fp16_t *>(k_param_->fil_ptr) + dz * weight_z_step;
                GemmFp16SlidewC3(dst_z, work_space_t, weight_dz, k_param_->ow, 
                                 conv_param->strides[0] * 8, kernel_x, kernel_y, dilate_x_step, src_xc * 8);
            }
        }
    }

    PostExec<fp16_t>(outputs);

    return TNN_OK;
}

}  // namespace TNN_NS
#endif
