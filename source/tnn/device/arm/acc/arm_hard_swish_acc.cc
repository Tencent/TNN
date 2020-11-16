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

#include "tnn/device/arm/acc/arm_hard_swish_acc.h"
#include "arm_binary_layer_acc.h"
#include "tnn/device/arm/acc/Float4.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

static inline Float4 SwishElement(Float4 v0, Float4 v1, float alpha, float beta) {
    return v0 * Float4::max(Float4::min(v1 * alpha + beta, 1.0f), 0.f);
}

static void HardSwishSingle(const float *src, float *dst, int count, float alpha, float beta) {
    OMP_PARALLEL_FOR_
    for (int n = 0; n < UP_DIV(count, 4); n++) {
        Float4 val = Float4::load(src + n * 4);
        Float4::save(dst + n * 4, SwishElement(val, val, alpha, beta));
    }
}

ArmHardSwishLayerAcc::~ArmHardSwishLayerAcc() {}
Status ArmHardSwishLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                  const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status status = ArmBinaryLayerAcc::Init(context, param, resource, inputs, outputs);
    if (status != TNN_OK) {
        return status;
    }

    auto layer_param = dynamic_cast<HardSwishLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    _Operator = [=](Float4 v1, Float4 v2) -> Float4 {
        Float4 dst = SwishElement(v1, v2, layer_param->alpha, layer_param->beta);
        return dst;
    };

    return TNN_OK;
}

Status ArmHardSwishLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<HardSwishLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);

    auto dims = outputs[0]->GetBlobDesc().dims;
    int count = dims[0] * ROUND_UP(dims[1], 4) * dims[2] * dims[3];

    float *output_data = reinterpret_cast<float *>(GetBlobHandlePtr(outputs[0]->GetHandle()));
    float *input_data  = reinterpret_cast<float *>(GetBlobHandlePtr(inputs[0]->GetHandle()));
    if (inputs.size() == 1) {
        HardSwishSingle(input_data, output_data, count, layer_param->alpha, layer_param->beta);
    } else {
        return ArmBinaryLayerAcc::DoForward(inputs, outputs);
    }
    return TNN_OK;
}

REGISTER_ARM_ACC(HardSwish, LAYER_HARDSWISH);

}  // namespace TNN_NS
