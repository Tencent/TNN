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

#include <algorithm>
#include <cmath>

#include "tnn/device/cpu/acc/cpu_layer_acc.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_CPU_ACC(GLU, LAYER_GLU);

Status CpuGLULayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuGLULayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto params = dynamic_cast<GLULayerParam *>(param_);
    if (!params) {
        LOGE("Error: GLULayerParam is unsupported\n");
        return Status(TNNERR_MODEL_ERR, "Error: GLULayerParam is unsupported");
    }

    Blob *input_blob        = inputs[0];
    Blob *output_blob       = outputs[0];
    float *input_ptr        = static_cast<float *>(input_blob->GetHandle().base);
    float *output_ptr       = static_cast<float *>(output_blob->GetHandle().base);
    const auto &input_dims  = input_blob->GetBlobDesc().dims;
    const auto &output_dims = output_blob->GetBlobDesc().dims;
    const int axis          = params->axis;
    const int batch         = DimsVectorUtils::Count(input_dims, 0, axis);
    const int input_split   = input_dims[axis];
    const int output_split  = output_dims[axis];
    const int count         = DimsVectorUtils::Count(input_dims, axis + 1);

    for (int n = 0; n < batch; n++) {
        float *const input_batch  = input_ptr + n * input_split * count;
        float *const output_batch = output_ptr + n * output_split * count;
        for (int c = 0; c < output_split; ++c) {
            // first half
            float *first_half_ptr = input_batch + c * count;
            // second half
            float *second_half_ptr = input_batch + (output_split + c) * count;
            float *output_data     = output_batch + c * count;
            for (int i = 0; i < count; ++i) {
                output_data[i] = first_half_ptr[i] / (1.0f + std::exp(-second_half_ptr[i]));
            }
        }
    }
    return TNN_OK;
}

REGISTER_CPU_ACC(GLU, LAYER_GLU);

}  // namespace TNN_NS
