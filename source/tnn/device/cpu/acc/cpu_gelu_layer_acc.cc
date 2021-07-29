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
#include "tnn/utils/bfp16.h"
#include "tnn/utils/bfp16_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_CPU_ACC(Gelu, LAYER_GELU);

Status CpuGeluLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuGeluLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    //GELU(x) = 0.5f * x * (erf(x*0.707106793288165f) + 1.0f); use erf in math.h or see the func in cpu_erf_layer_acc.cc
    //GELU(x) = 0.5f * x * (tanh((x+x*x*x*0.0447149984538f)*0.7978845834732056f)+1.0f), the approximation has big error if input is -2.281006575
    
    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];
    int count         = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims);
    auto data_type    = output_blob->GetBlobDesc().data_type;
    if (data_type == DATA_TYPE_FLOAT) {
        float *input_data  = static_cast<float *>(input_blob->GetHandle().base);
        float *output_data = static_cast<float *>(output_blob->GetHandle().base);
        for (int index = 0; index < count; ++index) {
            auto x = input_data[index];
            output_data[index] = 0.5f * x * (erf(x*0.707106793288165f) + 1.0f);
        }
    } else {
        LOGE("CpuGeluLayerAcc dont support data type: %d", data_type);
        return Status(TNNERR_NO_RESULT, "CpuGeluLayerAcc dont support data type");
    }
    return TNN_OK;
}

REGISTER_CPU_ACC(Gelu, LAYER_GELU);

}  // namespace TNN_NS
