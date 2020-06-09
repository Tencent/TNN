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

#include "tnn/device/arm/acc/convolution/arm_conv_layer_c3.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {
// usually appears on the first conv layer
bool ArmConvLayerC3::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                const std::vector<Blob *> &outputs) {
    if (!param) {
        return false;
    }
    return inputs[0]->GetBlobDesc().dims[1] == 3 && param->group == 1;
}

ArmConvLayerC3::~ArmConvLayerC3() {}

Status ArmConvLayerC3::allocateBufferWeight(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);
    ConvLayerResource *conv_res = dynamic_cast<ConvLayerResource *>(resource_);
    CHECK_PARAM_NULL(conv_res);

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;

    if (!buffer_weight_.GetBytesSize()) {
        const int input_channel  = dims_input[1];
        const int output_channel = dims_output[1];
        const int oc_4           = UP_DIV(output_channel, 4);
        const int ic_4           = UP_DIV(input_channel, 4);

        int kw = conv_param->kernels[0];
        int kh = conv_param->kernels[1];

        int weight_count   = oc_4 * ic_4 * kh * kw * 16;
        int data_byte_size = DataTypeUtils::GetBytesSize(conv_res->filter_handle.GetDataType());

        buffer_weight_   = RawBuffer(weight_count * data_byte_size);
        const float *src = conv_res->filter_handle.force_to<float *>();
        float *dst       = buffer_weight_.force_to<float *>();

        ConvertWeightsFromOI3HWToOHW12((float *)src, (float *)dst, input_channel, output_channel,
                                       conv_param->kernels[1], conv_param->kernels[0]);
    }
    return TNN_OK;
}
template <typename T>
void GemmSlidewC3(T *dst, const T *src, const float *weight, int width, int src_w_setup, int fw, int fh,
                  int dilateX_step, int dilateY_step) {
    LOGE("TYPE NOT IMPLEMENT");
}

template <>
void GemmSlidewC3(float *dst, const float *src, const float *weight, int width, int src_w_setup, int fw, int fh,
                  int dilateX_step, int dilateY_step) {
    GemmFloatSlidewC3(dst, src, weight, width, src_w_setup, fw, fh, dilateX_step, dilateY_step);
}

template <>
void GemmSlidewC3(bfp16_t *dst, const bfp16_t *src, const float *weight, int width, int src_w_setup, int fw, int fh,
                  int dilateX_step, int dilateY_step) {
    GemmBfp16SlidewC3(dst, src, weight, width, src_w_setup, fw, fh, dilateX_step, dilateY_step);
}

template <typename T>
Status ArmConvLayerC3::Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
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
    int dilate_y_step          = k_param_->iw * 4 * conv_param->dialations[1];
    int dilate_x_step          = 4 * conv_param->dialations[0];

    int weight_z_step = kernel_y * kernel_x * 12;

    T *src_origin = reinterpret_cast<T *>(GetBlobHandlePtr(input->GetHandle()));
    T *dst_origin = reinterpret_cast<T *>(GetBlobHandlePtr(output->GetHandle()));

    int max_num_threads = OMP_MAX_THREADS_NUM_;

    int src_xc = 1 + (k_param_->ow - 1) * conv_param->strides[0] + conv_param->dialations[0] * (kernel_x - 1);
    int workspace_per_thread = src_xc * kernel_y * k_param_->ic_r4 * data_byte_size;
    T *work_space = reinterpret_cast<T *>(context_->GetSharedWorkSpace(max_num_threads * workspace_per_thread));

    for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
        auto input_ptr  = src_origin + batch_idx * k_param_->iw * k_param_->ih * k_param_->ic_r4;
        auto output_ptr = dst_origin + batch_idx * k_param_->ow * k_param_->oh * k_param_->oc_r4;
        int src_start_x = 0 - conv_param->pads[0];
        int src_end_x   = src_start_x + src_xc >= k_param_->iw ? k_param_->iw : src_start_x + src_xc;

        int dst_offset = 0;
        if (src_start_x < 0) {
            dst_offset  = -src_start_x;
            src_start_x = 0;
        }
        int copy_count = src_end_x - src_start_x;
        auto src_x     = input_ptr + 4 * src_start_x;

        OMP_PARALLEL_FOR_
        for (int dy = 0; dy < k_param_->oh; dy++) {
            int thread_id = OMP_TID_;

            auto work_space_t = work_space + thread_id * workspace_per_thread / sizeof(T);
            memset(work_space_t, 0, workspace_per_thread);
            int src_start_y = dy * conv_param->strides[1] - conv_param->pads[2];
            int sfy         = MAX(0, (UP_DIV(-src_start_y, conv_param->dialations[1])));
            int efy         = MIN(kernel_y, UP_DIV(k_param_->ih - src_start_y, conv_param->dialations[1]));

            // copy make board
            for (int ky = sfy; ky < efy; ky++) {
                int sy     = src_start_y + ky * conv_param->dialations[1];
                auto src_y = src_x + 4 * sy * k_param_->iw;
                auto dst_y = work_space_t + (ky * src_xc + dst_offset) * 4;
                memcpy(dst_y, src_y, copy_count * 4 * data_byte_size);
            }
            for (int dz = 0; dz < k_param_->oc_r4 / 4; dz++) {
                auto dst_z =
                    reinterpret_cast<T *>(output_ptr) + dz * k_param_->ow * k_param_->oh * 4 + k_param_->ow * 4 * dy;
                auto weight_dz = reinterpret_cast<float *>(k_param_->fil_ptr) + dz * weight_z_step;
                // process one line at a time
                GemmSlidewC3(dst_z, reinterpret_cast<T *>(work_space_t), weight_dz, k_param_->ow,
                             conv_param->strides[0] * 4, kernel_x, kernel_y, dilate_x_step, src_xc * 4);
            }
        }
    }

    PostExec<T>(outputs);

    return TNN_OK;
}

Status ArmConvLayerC3::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        return Exec<float>(inputs, outputs);
    } else if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_BFP16) {
        return Exec<bfp16_t>(inputs, outputs);
    }
    return TNNERR_LAYER_ERR;
}

}  // namespace TNN_NS
