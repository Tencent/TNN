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

#include "tnn/device/arm/acc/arm_layer_acc.h"

#include <cmath>

#include "tnn/device/arm/acc/Float4.h"
#include "tnn/device/arm/arm_device.h"

namespace TNN_NS {

DECLARE_ARM_ACC(Pow, LAYER_POWER);

Status ArmPowLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<PowLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);

    Blob *output_blob = outputs[0];
    auto dims         = output_blob->GetBlobDesc().dims;
    int count         = dims[0] * ROUND_UP(dims[1], 4) * DimsVectorUtils::Count(dims, 2);
    int count_quad    = UP_DIV(count, 4);

    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        float *input_data  = reinterpret_cast<float *>(GetBlobHandlePtr(inputs[0]->GetHandle()));
        float *output_data = reinterpret_cast<float *>(GetBlobHandlePtr(output_blob->GetHandle()));

        int pow = round(layer_param->exponent);

        if (ABS(pow - layer_param->exponent) < FLT_EPSILON) {
            if (pow == 0) {
                for (int n = 0; n < count_quad; n++) {
                    Float4::save(output_data + n * 4, Float4(1.0f));
                }
                return TNN_OK;
            }

            bool reciprocal = pow < 0;
            if(reciprocal)
                pow = -pow;
            for (int n = 0; n < count_quad; n++) {
                Float4 val = Float4::load(input_data + n * 4) * layer_param->scale + layer_param->shift;
                if (reciprocal) {
                    val = Float4::div(1.0f, val);
                }
                Float4 res = val;
                for (int i = 0; i < pow - 1; i++) {
                    res = res * val;
                }
                Float4::save(output_data + n * 4, res);
            }
        } else {
            for (int n = 0; n < count_quad; n++) {
                Float4 val = Float4::load(input_data + n * 4);
                Float4 res =
                    Float4::pow(val * layer_param->scale + Float4(layer_param->shift), Float4(layer_param->exponent));
                Float4::save(output_data + n * 4, res);
            }
        }
    } else if (output_blob->GetBlobDesc().data_type == DATA_TYPE_INT8) {
        return Status(TNNERR_MODEL_ERR, "Error: layer acc dont support datatype");
    } else {
        return Status(TNNERR_MODEL_ERR, "Error: layer acc dont support datatype");
    }

    return TNN_OK;
}

REGISTER_ARM_ACC(Pow, LAYER_POWER);
REGISTER_ARM_LAYOUT(LAYER_POWER, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
