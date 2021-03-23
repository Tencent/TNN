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
#include "tnn/device/arm/arm_device.h"

#include "tnn/device/arm/acc/Float4.h"

namespace TNN_NS {

DECLARE_ARM_ACC(Sign, LAYER_SIGN);

Status ArmSignLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto dims      = outputs[0]->GetBlobDesc().dims;
    int count      = dims[0] * ROUND_UP(dims[1], 4) * DimsVectorUtils::Count(dims, 2);
    int count_quad = UP_DIV(count, 4);

    if (outputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        float *input_data  = reinterpret_cast<float *>(GetBlobHandlePtr(inputs[0]->GetHandle()));
        float *output_data = reinterpret_cast<float *>(GetBlobHandlePtr(outputs[0]->GetHandle()));
        Float4 zero        = Float4(0.f);
        Float4 one         = Float4(1.f);
        Float4 neg_one     = Float4(-1.f);
        for (int n = 0; n < count_quad; n++) {
            Float4 val = Float4::load(input_data + n * 4);
            Float4 res = Float4::bsl_clt(val, zero, neg_one, val);
            res        = Float4::bsl_cgt(val, zero, one, res);
            Float4::save(output_data + n * 4, res);
        }
    } else {
        return Status(TNNERR_MODEL_ERR, "Error: layer acc dont support datatype");
    }
    return TNN_OK;
}

REGISTER_ARM_ACC(Sign, LAYER_SIGN);
REGISTER_ARM_LAYOUT(LAYER_SIGN, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
