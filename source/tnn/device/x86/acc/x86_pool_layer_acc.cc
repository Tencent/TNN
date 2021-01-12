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

#include "tnn/device/x86/acc/compute/x86_compute.h"

namespace TNN_NS {

DECLARE_X86_ACC(Pool, LAYER_POOLING);

Status X86PoolLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status X86PoolLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<PoolingLayerParam *>(param_);
    if (!param) {
        return Status(TNNERR_MODEL_ERR, "Error: PoolingLayerParam is nil");
    }

    int stride_w = param->strides[0];
    int stride_h = param->strides[1];
    int pad_w    = param->pads[0];
    int pad_h    = param->pads[2];
    int kernel_w = param->kernels[0];
    int kernel_h = param->kernels[1];
    auto pool_type = param->pool_type;

    auto input = inputs[0];
    auto output = outputs[0];

    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;

    if (output->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        if (pool_type == 0) {
            X86_MAX_POOLING(reinterpret_cast<float *>(input->GetHandle().base),
                            reinterpret_cast<float *>(output->GetHandle().base),
                            dims_input, dims_output, stride_h, stride_w,
                            kernel_h, kernel_w, pad_h, pad_w);
        } else {
            X86_AVERAGE_POOLING(reinterpret_cast<float *>(input->GetHandle().base),
                                reinterpret_cast<float *>(output->GetHandle().base),
                                dims_input, dims_output, stride_h, stride_w,
                                kernel_h, kernel_w, pad_h, pad_w);
        }
    } else {
        return Status(TNNERR_DEVICE_ACC_DATA_FORMAT_NOT_SUPPORT, "Error: this data type not supported in pooling layer");
    }

    return TNN_OK;
}

REGISTER_X86_ACC(Pool, LAYER_POOLING);
}