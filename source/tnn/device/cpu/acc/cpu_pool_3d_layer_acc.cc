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

#include "tnn/utils/naive_compute.h"
#include "tnn/device/cpu/acc/cpu_layer_acc.h"
#include "tnn/utils/bfp16.h"

namespace TNN_NS {

DECLARE_CPU_ACC(Pool3D, LAYER_POOLING_3D);

Status CpuPool3DLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuPool3DLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<PoolingLayerParam *>(param_);
    if (!param) {
        return Status(TNNERR_MODEL_ERR, "Error: PoolingLayerParam is nil");
    }
    if (inputs[0]->GetBlobDesc().data_format != DATA_FORMAT_NCDHW) {
        LOGE("Error: Pool3D layer only support NCDHW data format\n");
        return Status(TNNERR_LAYER_ERR, "Error: Pool3D layer only support NCDHW data format");
    }
    if (outputs[0]->GetBlobDesc().data_format != DATA_FORMAT_NCDHW) {
        LOGE("Error: Pool3D layer only support NCDHW data format\n");
        return Status(TNNERR_LAYER_ERR, "Error: Pool3D layer only support NCDHW data format");
    }

    int stride_x   = param->strides[0];
    int stride_y   = param->strides[1];
    int stride_d   = param->strides[2];
    int pad_x      = param->pads[0];
    int pad_y      = param->pads[2];
    int pad_d      = param->pads[4];
    int kernel_x   = param->kernels[0];
    int kernel_y   = param->kernels[1];
    int kernel_d   = param->kernels[2];
    auto pool_type = param->pool_type;

    auto input  = inputs[0];
    auto output = outputs[0];

    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;

    if (output->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        NaivePooling3D<float, float>(reinterpret_cast<float *>(input->GetHandle().base),
                                    reinterpret_cast<float *>(output->GetHandle().base), dims_input, dims_output,
                                    stride_d, stride_y, stride_x, kernel_d, kernel_y, kernel_x,
                                    pad_d, pad_y, pad_x, pool_type);
    } else if (output->GetBlobDesc().data_type == DATA_TYPE_BFP16) {
        NaivePooling3D<bfp16_t, float>(reinterpret_cast<bfp16_t *>(input->GetHandle().base),
                                    reinterpret_cast<bfp16_t *>(output->GetHandle().base), dims_input, dims_output,
                                    stride_d, stride_y, stride_x, kernel_d, kernel_y, kernel_x,
                                    pad_d, pad_y, pad_x, pool_type);
    } else if (output->GetBlobDesc().data_type == DATA_TYPE_INT8) {
        NaivePooling3D<int8_t, int32_t>(reinterpret_cast<int8_t *>(input->GetHandle().base),
                                    reinterpret_cast<int8_t *>(output->GetHandle().base), dims_input, dims_output,
                                    stride_d, stride_y, stride_x, kernel_d, kernel_y, kernel_x,
                                    pad_d, pad_y, pad_x, pool_type);
    }

    return TNN_OK;
}

REGISTER_CPU_ACC(Pool3D, LAYER_POOLING_3D);

}  // namespace TNN_NS
