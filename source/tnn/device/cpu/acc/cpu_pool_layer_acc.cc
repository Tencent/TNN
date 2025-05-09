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

DECLARE_CPU_ACC_WITH_FUNC(Pool, LAYER_POOLING,
                          virtual Status InferRuntimeOutputShape(const std::vector<Blob *> &inputs,
                                                                 const std::vector<Blob *> &outputs););

Status CpuPoolLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuPoolLayerAcc::InferRuntimeOutputShape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<PoolingLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);

    auto input_dims = inputs[0]->GetBlobDesc().dims;
    if (layer_param->is_adaptive_pool && inputs.size() > 1) {
        if (inputs[1]->GetBlobDesc().data_type != DATA_TYPE_INT32) {
            return Status(TNNERR_PARAM_ERR, "adaptive pool input(shape) has invalid data type");
        }
        auto dim_count = DimsVectorUtils::Count(inputs[1]->GetBlobDesc().dims);
        auto dim_data = (int *)((char *)inputs[1]->GetHandle().base + inputs[1]->GetHandle().bytes_offset);
        layer_param->output_shape.clear();
        for (int i = dim_count - 1; i >= 0; i--) {
            layer_param->output_shape.push_back(dim_data[i]);
        }
        std::vector<int> output_dims = {input_dims[0], input_dims[1], layer_param->output_shape[1], layer_param->output_shape[0]};
        outputs[0]->GetBlobDesc().dims = output_dims;
    }

    return TNN_OK;
}

Status CpuPoolLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<PoolingLayerParam *>(param_);
    if (!param) {
        return Status(TNNERR_MODEL_ERR, "Error: PoolingLayerParam is nil");
    }

    int stride_x   = param->strides[0];
    int stride_y   = param->strides[1];
    int pad_x      = param->pads[0];
    int pad_y      = param->pads[2];
    int kernel_x   = param->kernels[0];
    int kernel_y   = param->kernels[1];
    auto pool_type = param->pool_type;

    auto input  = inputs[0];
    auto output = outputs[0];

    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;

    auto input_width = dims_input[3], input_height = dims_input[2];
    auto output_width = dims_output[3], output_height = dims_output[2], output_channel = dims_output[1];

    if (output->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        if (param->is_adaptive_pool) {
            NaiveAdaptivePooling<float, float>(reinterpret_cast<float *>(input->GetHandle().base),
                                               reinterpret_cast<float *>(output->GetHandle().base), dims_input,
                                               dims_output, pool_type);
        } else {
            NaivePooling<float, float>(reinterpret_cast<float *>(input->GetHandle().base),
                                       reinterpret_cast<float *>(output->GetHandle().base), dims_input, dims_output,
                                       stride_y, stride_x, kernel_y, kernel_x, pad_y, pad_x, pool_type);
        }
    } else if (output->GetBlobDesc().data_type == DATA_TYPE_BFP16) {
        NaivePooling<bfp16_t, float>(reinterpret_cast<bfp16_t *>(input->GetHandle().base),
                                    reinterpret_cast<bfp16_t *>(output->GetHandle().base), dims_input, dims_output,
                                    stride_y, stride_x, kernel_y, kernel_x, pad_y, pad_x, pool_type);
    } else if (output->GetBlobDesc().data_type == DATA_TYPE_INT8) {
        NaivePooling<int8_t, int32_t>(reinterpret_cast<int8_t *>(input->GetHandle().base),
                                    reinterpret_cast<int8_t *>(output->GetHandle().base), dims_input, dims_output,
                                    stride_y, stride_x, kernel_y, kernel_x, pad_y, pad_x, pool_type);
    }

    return TNN_OK;
}

REGISTER_CPU_ACC(Pool, LAYER_POOLING);

}  // namespace TNN_NS
