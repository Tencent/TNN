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
#include "tnn/device/cpu/unpack_nchw/acc/nchw_layer_acc.h"
#include "tnn/device/cpu/unpack_nchw/nchw_device.h"
#include "tnn/device/cpu/unpack_nchw/nchw_common.h"
#include "tnn/device/cpu/unpack_nchw/nchw_util.h"

#include "tnn/device/cpu/unpack_nchw/acc/compute/nchw_compute.h"
#include "tnn/device/cpu/unpack_nchw/acc/nchw_pool_layer_acc.h"
#include "tnn/device/cpu/unpack_nchw/Float8.h"
#include "tnn/device/cpu/unpack_nchw/Float4.h"

namespace TNN_NS {

NchwPoolLayerAcc::~NchwPoolLayerAcc() {}

Status NchwPoolLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
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

template <typename dtype, typename VEC, int pack>
Status NchwPoolLayerAcc::DoForwardImpl(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
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
    auto input_ptr  = static_cast<dtype *>(input->GetHandle().base);
    auto output_ptr = static_cast<dtype *>(output->GetHandle().base);

    auto NchwMaxPoolingAcc = NchwMaxPooling<VEC, pack>;
    auto NchwAvgPoolingAcc = NchwAvgPooling<VEC, pack>;
    auto PackAcc          = Pack<VEC, pack>;
    auto UnpackAcc        = Unpack<VEC, pack>;
    int c_pack = pack;

    size_t src_hw        = dims_input[3] * dims_input[2];
    size_t dst_hw        = dims_output[3] * dims_output[2];
    size_t src_pack_size = ROUND_UP(src_hw * c_pack * sizeof(dtype), 32);
    size_t dst_pack_size = ROUND_UP(dst_hw * c_pack * sizeof(dtype), 32);
    dtype *workspace = reinterpret_cast<dtype *>(context_->GetSharedWorkSpace(src_pack_size + dst_pack_size));
    auto src_pack_ptr = workspace;
    auto dst_pack_ptr = workspace + src_pack_size / sizeof(dtype);

    //OMP_PARALLEL_FOR_
    for (int b = 0; b < batch; b++) {
        auto input_b  = reinterpret_cast<dtype *>(input_ptr) + b * dims_input[1] * src_hw;
        auto output_b = reinterpret_cast<dtype *>(output_ptr) + b * dims_output[1] * dst_hw;
        for (int c = 0; c < dims_output[1]; c += c_pack) {
            int left_c = MIN(dims_output[1] - c, c_pack);
            PackAcc(src_pack_ptr, input_b + c * src_hw, src_hw, src_hw, src_hw, left_c);
            if (param->pool_type == 0) {
                NchwMaxPoolingAcc(src_pack_ptr, dims_input[3], dims_input[2], dst_pack_ptr,
                        dims_output[3], dims_output[2], param->kernels[0], param->kernels[1], param->strides[0],
                        param->strides[1], param->pads[0], param->pads[2], corner_l_, corner_r_, corner_t_,
                        corner_b_);
            } else {
                NchwAvgPoolingAcc(src_pack_ptr, dims_input[3], dims_input[2], dst_pack_ptr,
                        dims_output[3], dims_output[2], param->kernels[0], param->kernels[1], param->strides[0],
                        param->strides[1], param->pads[0], param->pads[2]);
            }
            UnpackAcc(output_b + c * dst_hw, dst_pack_ptr, dst_hw, dst_hw, dst_hw, left_c);
        }
    }

    return TNN_OK;
}

Status NchwPoolLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
#if TNN_X86
    if (output->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        if (arch_ == avx2) {
            RETURN_ON_NEQ(DoForwardImpl<float, Float8, 8>(inputs, outputs), TNN_OK);
        } else if (arch_ == sse42) {
            RETURN_ON_NEQ(DoForwardImpl<float, Float4, 4>(inputs, outputs), TNN_OK);
        }
    } else {
        return Status(TNNERR_DEVICE_ACC_DATA_FORMAT_NOT_SUPPORT, "Error: this data type not supported in pooling layer");
    }
#else if TNN_ARM
    if (output->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        RETURN_ON_NEQ(DoForwardImpl<float, Float4, 4>(inputs, outputs), TNN_OK);
    } else if (output->GetBlobDesc().data_type == DATA_TYPE_HALF) {
        RETURN_ON_NEQ(DoForwardImpl<fp16_t, Half8, 8>(inputs, outputs), TNN_OK);
    } else {
        return Status(TNNERR_DEVICE_ACC_DATA_FORMAT_NOT_SUPPORT, "Error: this data type not supported in pooling layer");
    }
#endif
}

REGISTER_NCHW_ACC(Pool, LAYER_POOLING);
}