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

#include <graph/op/all_ops.h>
#include "npu_unary_operator.h"
#include "tnn/device/huawei_npu/convert/npu_base_layer_convert.h"
#include "tnn/device/huawei_npu/convert/npu_utils.h"

namespace TNN_NS {

class NpuHardswishLayer : public NpuUnaryLayer {
public:
    NpuHardswishLayer(LayerType ignore) : NpuUnaryLayer(LAYER_HARDSWISH) {}
    ~NpuHardswishLayer() {}

protected:
    Status Convert() {
        auto param = dynamic_cast<HardSwishLayerParam *>(param_);
        if (!(param->alpha >= 0.1666f && param->alpha <= 0.1667f && param->beta >= 0.4999f && param->beta <= 0.5001f)) {
            LOGE("hardswish only support alpha=1/6 beta=0.5, but in fact, alpha=%f beta=%f\n", param->alpha, param->beta);
            return Status(TNNERR_LAYER_ERR, "Error: Npu currently only supports hardswish (alpha=1/6, beta=0.5)");
        }
        return NpuUnaryLayer::UnaryConvert<hiai::op::HardSwish>();
    }
};

REGISTER_NPU_LAYER(Hardswish, LAYER_HARDSWISH);

}  // namespace TNN_NS
