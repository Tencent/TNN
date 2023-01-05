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

#include "tnn/core/blob.h"
#include "tnn/device/x86/acc/x86_layer_acc.h"
#include "tnn/device/x86/x86_device.h"
#include "tnn/device/x86/x86_common.h"
#include "tnn/device/x86/x86_util.h"
#include "tnn/utils/omp_utils.h"

#include "tnn/device/x86/acc/compute/x86_compute.h"
#include "tnn/device/x86/acc/compute/x86_compute_int8.h"
#include "tnn/device/x86/acc/x86_pool_layer_acc.h"
#include "tnn/device/x86/acc/Float8.h"
#include "tnn/device/x86/acc/Float4.h"

namespace TNN_NS {
using namespace x86;

X86PoolLayerAcc::~X86PoolLayerAcc() {}

Status X86PoolLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    PoolingLayerParam *param = dynamic_cast<PoolingLayerParam *>(param_);
    CHECK_PARAM_NULL(param);
    auto input = inputs[0];
    auto output = outputs[0];
    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;

    int corner_l_ = 0, corner_t_ = 0, corner_r_ = dims_output[3], corner_b_ = dims_output[2];
    for (; corner_l_ * param->strides[0] - param->pads[0] < 0; corner_l_++)
        ;
    for (; corner_t_ * param->strides[1] - param->pads[2] < 0; corner_t_++)
        ;
    for (; (corner_r_ - 1) * param->strides[0] - param->pads[0] + param->kernels[0] > dims_input[3] &&
            corner_r_ > corner_l_;
        corner_r_--)
        ;
    for (; (corner_b_ - 1) * param->strides[1] - param->pads[2] + param->kernels[1] > dims_input[2] &&
            corner_b_ > corner_t_;
        corner_b_--)
        ;
    return TNN_OK;
}

Status X86PoolLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<PoolingLayerParam *>(param_);
    if (!param) {
        return Status(TNNERR_MODEL_ERR, "Error: PoolingLayerParam is nil");
    }

    auto pool_type = param->pool_type;

    auto input = inputs[0];
    auto output = outputs[0];

    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;

    auto batch      = dims_output[0];
    auto oc_r4      = ROUND_UP(dims_output[1], 4);
    auto input_ptr  = handle_ptr<float *>(input->GetHandle());
    auto output_ptr = handle_ptr<float *>(output->GetHandle());

    auto X86MaxPoolingAcc = X86MaxPooling<Float4, 4>;
    auto X86AvgPoolingAcc = X86AvgPooling<Float4, 4>;
    auto PackAcc          = PackC4;
    auto UnpackAcc        = UnpackC4;
    int c_pack = 4;
    if (arch_ == avx2) {
        X86MaxPoolingAcc = X86MaxPooling<Float8, 8>;
        X86AvgPoolingAcc = X86AvgPooling<Float8, 8>;
        PackAcc          = PackC8;
        UnpackAcc        = UnpackC8;
        c_pack = 8;
    }

    int max_num_threads  = OMP_MAX_THREADS_NUM_;
    size_t src_hw        = dims_input[3] * dims_input[2];
    size_t dst_hw        = dims_output[3] * dims_output[2];
    size_t src_pack_size = ROUND_UP(src_hw * c_pack * sizeof(float), 32);
    size_t dst_pack_size = ROUND_UP(dst_hw * c_pack * sizeof(float), 32);
    float *workspace = reinterpret_cast<float *>(context_->GetSharedWorkSpace(
                        (src_pack_size + dst_pack_size) * max_num_threads));

    if (output->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        for (int b = 0; b < batch; b++) {
            auto input_b  = reinterpret_cast<float *>(input_ptr) + b * dims_input[1] * src_hw;
            auto output_b = reinterpret_cast<float *>(output_ptr) + b * dims_output[1] * dst_hw;
            OMP_PARALLEL_FOR_GUIDED_
            for (int c = 0; c < dims_output[1]; c += c_pack) {
                int thread_id = OMP_TID_;
                auto workspace_per_t = workspace + thread_id * ((src_pack_size + dst_pack_size) / sizeof(float));
                auto src_pack_ptr    = workspace_per_t;
                auto dst_pack_ptr    = workspace_per_t + src_pack_size / sizeof(float);
                int left_c = MIN(dims_output[1] - c, c_pack);
                PackAcc(src_pack_ptr, input_b + c * src_hw, src_hw, src_hw, src_hw, left_c);
                if (param->pool_type == 0) {
                    X86MaxPoolingAcc(src_pack_ptr, dims_input[3], dims_input[2], dst_pack_ptr,
                            dims_output[3], dims_output[2], param->kernels[0], param->kernels[1], param->strides[0],
                            param->strides[1], param->pads[0], param->pads[2], corner_l_, corner_r_, corner_t_,
                            corner_b_);
                } else {
                    X86AvgPoolingAcc(src_pack_ptr, dims_input[3], dims_input[2], dst_pack_ptr,
                            dims_output[3], dims_output[2], param->kernels[0], param->kernels[1], param->strides[0],
                            param->strides[1], param->pads[0], param->pads[2]);
                }
                UnpackAcc(output_b + c * dst_hw, dst_pack_ptr, dst_hw, dst_hw, dst_hw, left_c);
            }
        }
    } else if (input->GetBlobDesc().data_type == DATA_TYPE_INT8) {
        // INT8
        for (int n = 0; n < batch; n++) {
            auto input_batch_stride  = dims_input[3] * dims_input[2] * oc_r4;
            auto output_batch_stride = dims_output[3] * dims_output[2] * oc_r4;
            if (param->pool_type == 0) {
                X86MaxPoolingINT8(reinterpret_cast<int8_t *>(input_ptr) + n * input_batch_stride, dims_input[3],
                               dims_input[2], reinterpret_cast<int8_t *>(output_ptr) + n * output_batch_stride,
                               dims_output[3], dims_output[2], oc_r4, param->kernels[0], param->kernels[1],
                               param->strides[0], param->strides[1], param->pads[0], param->pads[2]);
            } else {
                X86AvgPoolingINT8(reinterpret_cast<int8_t *>(input_ptr) + n * input_batch_stride, dims_input[3],
                               dims_input[2], reinterpret_cast<int8_t *>(output_ptr) + n * output_batch_stride,
                               dims_output[3], dims_output[2], oc_r4, param->kernels[0], param->kernels[1],
                               param->strides[0], param->strides[1], param->pads[0], param->pads[2]);
            }
        }
    } else {
        return Status(TNNERR_DEVICE_ACC_DATA_FORMAT_NOT_SUPPORT, "Error: this data type not supported in pooling layer");
    }

    return TNN_OK;
}

REGISTER_X86_ACC(Pool, LAYER_POOLING);
}
