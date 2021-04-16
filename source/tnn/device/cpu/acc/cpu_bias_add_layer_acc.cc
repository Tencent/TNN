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

#include "tnn/device/cpu/acc/cpu_layer_acc.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_CPU_ACC(BiasAdd, LAYER_BIAS_ADD);

Status CpuBiasAddLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuBiasAddLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto resource = dynamic_cast<BiasAddLayerResource *>(resource_);
    if (!resource) {
        return Status(TNNERR_MODEL_ERR, "Error: BiasAddLayerResource is nil");
    }

    auto input_blob        = inputs[0];
    auto output_blob       = outputs[0];
    float *input_data      = static_cast<float *>(input_blob->GetHandle().base);
    float *output_data     = static_cast<float *>(output_blob->GetHandle().base);
    auto dims              = output_blob->GetBlobDesc().dims;
    int batch              = dims[0];
    int channel            = dims[1];
    auto *bias_data        = resource->bias_handle.force_to<float *>();

    const int inner_size = DimsVectorUtils::Count(input_blob->GetBlobDesc().dims, 2);

    for (int b = 0; b < batch; ++b) {
        for(int c = 0; c < channel; ++c) {
            float bias = bias_data[c];
            for (int i = 0; i < inner_size; ++i, ++output_data, ++input_data) {
                *output_data = *input_data + bias;
            }
        }
    }

    return TNN_OK;
}

REGISTER_CPU_ACC(BiasAdd, LAYER_BIAS_ADD);

}  // namespace TNN_NS
