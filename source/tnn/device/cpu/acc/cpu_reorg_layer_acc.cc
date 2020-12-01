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

#include <cmath>

#include "tnn/device/cpu/acc/cpu_layer_acc.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/naive_compute.h"

namespace TNN_NS {

DECLARE_CPU_ACC(Reorg, LAYER_REORG);

Status CpuReorgLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuReorgLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];
    auto layer_param  = dynamic_cast<ReorgLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);

    int stride  = layer_param->stride;
    int forward = layer_param->forward;
    int mode    = layer_param->mode;

    if (input_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        float *bottom_data = static_cast<float *>(input_blob->GetHandle().base);
        float *top_data    = static_cast<float *>(output_blob->GetHandle().base);
        if (forward) {
            DimsVector input_dims = input_blob->GetBlobDesc().dims;
            int batch             = input_dims[0];
            int channel           = input_dims[1];
            int height            = input_dims[2];
            int width             = input_dims[3];
            NaiveReorg(bottom_data, width, height, channel, batch, stride, forward, mode, top_data);
        } else {
            DimsVector output_dims = output_blob->GetBlobDesc().dims;
            int batch              = output_dims[0];
            int channel            = output_dims[1];
            int height             = output_dims[2];
            int width              = output_dims[3];
            NaiveReorg(bottom_data, width, height, channel, batch, stride, forward, mode, top_data);
        }
    }

    return TNN_OK;
}

REGISTER_CPU_ACC(Reorg, LAYER_REORG);

}  // namespace TNN_NS
