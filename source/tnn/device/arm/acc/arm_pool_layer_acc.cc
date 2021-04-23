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

#include "tnn/device/arm/acc/arm_pool_layer_acc.h"

#include "tnn/device/arm/acc/arm_layer_acc.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

ArmPoolingLayerAcc::~ArmPoolingLayerAcc(){};

Status ArmPoolingLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ArmLayerAcc::Reshape(inputs, outputs);
    PoolingLayerParam *param = dynamic_cast<PoolingLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    corner_l_ = 0, corner_t_ = 0, corner_r_ = k_param_->ow, corner_b_ = k_param_->oh;
    for (; corner_l_ * param->strides[0] - param->pads[0] < 0; corner_l_++)
        ;
    for (; corner_t_ * param->strides[1] - param->pads[2] < 0; corner_t_++)
        ;
    for (; (corner_r_ - 1) * param->strides[0] - param->pads[0] + param->kernels[0] > k_param_->iw &&
            corner_r_ > corner_l_;
        corner_r_--)
        ;
    for (; (corner_b_ - 1) * param->strides[1] - param->pads[2] + param->kernels[1] > k_param_->ih &&
            corner_b_ > corner_t_;
        corner_b_--)
        ;
    return TNN_OK;
}

Status ArmPoolingLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    PoolingLayerParam *param = dynamic_cast<PoolingLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    auto input  = inputs[0];
    auto output = outputs[0];

    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;

    auto oc_4       = UP_DIV(dims_output[1], 4);
    auto batch      = dims_output[0];
    auto input_ptr  = GetBlobHandlePtr(input->GetHandle());
    auto output_ptr = GetBlobHandlePtr(output->GetHandle());

    // run
    if (input->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        auto input_plane_stride  = 4 * k_param_->iw * k_param_->ih;
        auto output_plane_stride = 4 * k_param_->ow * k_param_->oh;
        OMP_PARALLEL_FOR_
        for (int plane = (int)0; plane < batch * oc_4; plane++) {
            if (param->pool_type == 0) {
                MaxPooling(reinterpret_cast<float *>(input_ptr) + plane * input_plane_stride, k_param_->iw,
                           k_param_->ih, reinterpret_cast<float *>(output_ptr) + output_plane_stride * plane,
                           k_param_->ow, k_param_->oh, param->kernels[0], param->kernels[1], param->strides[0],
                           param->strides[1], param->pads[0], param->pads[2], corner_l_, corner_r_, corner_t_,
                           corner_b_);
            } else {
                AvgPooling(reinterpret_cast<float *>(input_ptr) + plane * input_plane_stride, k_param_->iw,
                           k_param_->ih, reinterpret_cast<float *>(output_ptr) + output_plane_stride * plane,
                           k_param_->ow, k_param_->oh, param->kernels[0], param->kernels[1], param->strides[0],
                           param->strides[1], param->pads[0], param->pads[2]);
            }
        }
    } else if (input->GetBlobDesc().data_type == DATA_TYPE_BFP16) {
        auto input_plane_stride  = 4 * k_param_->iw * k_param_->ih;
        auto output_plane_stride = 4 * k_param_->ow * k_param_->oh;
        OMP_PARALLEL_FOR_
        for (int plane = (int)0; plane < batch * oc_4; plane++) {
            if (param->pool_type == 0) {
                MaxPooling(reinterpret_cast<bfp16_t *>(input_ptr) + plane * input_plane_stride, k_param_->iw,
                           k_param_->ih, reinterpret_cast<bfp16_t *>(output_ptr) + output_plane_stride * plane,
                           k_param_->ow, k_param_->oh, param->kernels[0], param->kernels[1], param->strides[0],
                           param->strides[1], param->pads[0], param->pads[2], corner_l_, corner_r_, corner_t_,
                           corner_b_);
            } else {
                AvgPooling(reinterpret_cast<bfp16_t *>(input_ptr) + plane * input_plane_stride, k_param_->iw,
                           k_param_->ih, reinterpret_cast<bfp16_t *>(output_ptr) + output_plane_stride * plane,
                           k_param_->ow, k_param_->oh, param->kernels[0], param->kernels[1], param->strides[0],
                           param->strides[1], param->pads[0], param->pads[2]);
            }
        }
    }
#if TNN_ARM82
    else if (input->GetBlobDesc().data_type == DATA_TYPE_HALF) {
        auto oc_8       = UP_DIV(dims_output[1], 8);
        auto input_plane_stride  = 8 * k_param_->iw * k_param_->ih;
        auto output_plane_stride = 8 * k_param_->ow * k_param_->oh;
        OMP_PARALLEL_FOR_
        for (int plane = (int)0; plane < batch * oc_8; plane++) {
            if (param->pool_type == 0) {
                MaxPoolingHalf(reinterpret_cast<fp16_t *>(input_ptr) + plane * input_plane_stride, k_param_->iw,
                               k_param_->ih, reinterpret_cast<fp16_t *>(output_ptr) + output_plane_stride * plane,
                               k_param_->ow, k_param_->oh, param->kernels[0], param->kernels[1], param->strides[0],
                               param->strides[1], param->pads[0], param->pads[2]);
            } else {
                AvgPoolingHalf(reinterpret_cast<fp16_t *>(input_ptr) + plane * input_plane_stride, k_param_->iw,
                               k_param_->ih, reinterpret_cast<fp16_t *>(output_ptr) + output_plane_stride * plane,
                               k_param_->ow, k_param_->oh, param->kernels[0], param->kernels[1], param->strides[0],
                               param->strides[1], param->pads[0], param->pads[2]);
            }
        }
    }
#endif
    else if (input->GetBlobDesc().data_type == DATA_TYPE_INT8) {
        // INT8
        for (int n = 0; n < batch; n++) {
            auto input_batch_stride  = k_param_->iw * k_param_->ih * oc_4 * 4;
            auto output_batch_stride = k_param_->ow * k_param_->oh * oc_4 * 4;
            if (param->pool_type == 0) {
                MaxPoolingINT8(reinterpret_cast<int8_t *>(input_ptr) + n * input_batch_stride, k_param_->iw,
                               k_param_->ih, reinterpret_cast<int8_t *>(output_ptr) + n * output_batch_stride,
                               k_param_->ow, k_param_->oh, oc_4 * 4, param->kernels[0], param->kernels[1],
                               param->strides[0], param->strides[1], param->pads[0], param->pads[2]);
            } else {
                AvgPoolingINT8(reinterpret_cast<int8_t *>(input_ptr) + n * input_batch_stride, k_param_->iw,
                               k_param_->ih, reinterpret_cast<int8_t *>(output_ptr) + n * output_batch_stride,
                               k_param_->ow, k_param_->oh, oc_4 * 4, param->kernels[0], param->kernels[1],
                               param->strides[0], param->strides[1], param->pads[0], param->pads[2]);
            }
        }
    } else {
        return Status(TNNERR_LAYER_ERR, "Error: arm pooling layer got unsupported data type");
    }

    return TNN_OK;
}

REGISTER_ARM_ACC(Pooling, LAYER_POOLING)
REGISTER_ARM_PRECISION_FP16(LAYER_POOLING)
REGISTER_ARM_LAYOUT(LAYER_POOLING, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
