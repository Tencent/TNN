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

#include "tnn/device/arm/acc/arm_gelu_layer_acc.h"

#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS
{

Status ArmGeluLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs){
    auto input = inputs[0];
    auto output = outputs[0];

    // n, c, h, w
    auto dims = output->GetBlobDesc().dims;

    //             n     *     round_up(c,4)    *              h*w
    long count = dims[0] * ROUND_UP(dims[1], 4) * DimsVectorUtils::Count(dims, 2);

    auto &data_type = input->GetBlobDesc().data_type;

    if (data_type == DATA_TYPE_FLOAT) {
        auto dst = reinterpret_cast<float *>(GetBlobHandlePtr(output->GetHandle()));
        auto src = reinterpret_cast<float *>(GetBlobHandlePtr(input->GetHandle()));

        // https://arxiv.org/abs/1606.08415
        // approximate calculation
        float F0 = 0.7978845834732056f;  // sqrt(2/pi)
        float F1 = 0.0447149984538f;            // 
        Float4 vone(1.0f);

        for (long i = 0; i < count; i += 4) {
            auto x = Float4::load(src + i);
            // auto tmp = Float4::pow(x, vthree); // pow only support positive number now
            auto tmp = x * x;
            tmp = tmp * x;

            tmp = tmp * F1;
            tmp = tmp + x;
            tmp = tmp * F0;
            tmp = Float4::tanh(tmp);
            tmp = tmp + vone;
            tmp = tmp * 0.5f;
            tmp = tmp * x;

            Float4::save(dst + i, tmp);
        }
    } else {
        return TNNERR_LAYER_ERR;
    }
    return TNN_OK;
}

REGISTER_ARM_ACC(Gelu, LAYER_GELU)
REGISTER_ARM_LAYOUT(LAYER_GELU, DATA_FORMAT_NC4HW4)

} // namespace TNN_NS
