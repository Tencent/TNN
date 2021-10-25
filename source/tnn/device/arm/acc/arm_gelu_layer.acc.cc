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
#include "tnn/utils/dims_function_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

static Float4 fast_erf(Float4 x) {
    auto t = Float4::div(1.0f, Float4::abs(x) * 0.5f + Float4(1.0f));

    auto t_2 = t * t;
    auto t_3 = t_2 * t;
    auto t_4 = t_3 * t;
    auto t_5 = t_4 * t;
    auto t_6 = t_5 * t;
    auto t_7 = t_6 * t;
    auto t_8 = t_7 * t;
    auto t_9 = t_8 * t;

    auto v = t * Float4::exp(-x * x - Float4(1.26551223) + t * 1.00002368 + t_2 * 0.37409196 + t_3 * 0.09678418 -
                             t_4 * 0.18628806 + t_5 * 0.27886807 - t_6 * 1.13520398 + t_7 * 1.48851587 -
                             t_8 * 0.82215223 + t_9 * 0.17087277);

    return Float4::bsl_cge(x, Float4(0.0f), Float4(1.0f) - v, v - Float4(1.0f));
}

Status ArmGeluLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];

    auto dims = output->GetBlobDesc().dims;

    int count = DimsFunctionUtils::GetDim(dims, 0) * ROUND_UP(DimsFunctionUtils::GetDim(dims, 1), 4) *
                DimsVectorUtils::Count(dims, 2);

    auto &data_type = input->GetBlobDesc().data_type;

    if (data_type == DATA_TYPE_FLOAT) {
        auto dst = reinterpret_cast<float *>(GetBlobHandlePtr(output->GetHandle()));
        auto src = reinterpret_cast<float *>(GetBlobHandlePtr(input->GetHandle()));

        if (context_->GetPrecision() == PRECISION_HIGH || context_->GetPrecision() == PRECISION_NORMAL) {
            // 0.5f * x * (erf(x*0.707106793288165f) + 1.0f);
            for (int i = 0; i < count; i += 4) {
                Float4 x   = Float4::load(src + i);
                Float4 res = x * 0.707106793288165f;
                res        = fast_erf(res);
                res        = res + Float4(1.0f);
                res        = res * 0.5f;
                res        = res * x;
                Float4::save(dst + i, res);
            }
        } else {
            // https://arxiv.org/abs/1606.08415
            // precision is auto, use approximate calculation
            // Note the approximation has big error if input is -2.281006575
            float F0 = 0.7978845834732056f;
            float F1 = 0.0447149984538f;
            Float4 vone(1.0f);

            for (int i = 0; i < count; i += 4) {
                auto x = Float4::load(src + i);

                auto tmp = x * x;
                tmp      = tmp * x;

                tmp = tmp * F1;
                tmp = tmp + x;
                tmp = tmp * F0;
                tmp = Float4::tanh(tmp);
                tmp = tmp + vone;
                tmp = tmp * 0.5f;
                tmp = tmp * x;

                Float4::save(dst + i, tmp);
            }
        }
    } else {
        return TNNERR_LAYER_ERR;
    }
    return TNN_OK;
}

REGISTER_ARM_ACC(Gelu, LAYER_GELU)
REGISTER_ARM_LAYOUT(LAYER_GELU, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
