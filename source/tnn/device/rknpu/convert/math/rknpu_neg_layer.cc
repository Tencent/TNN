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

#include "rknpu_unary_operator.h"
#include "tnn/device/rknpu/convert/rknpu_base_layer.h"
#include "tnn/device/rknpu/convert/rknpu_utils.h"

namespace TNN_NS {

class RknpuNegLayer : public RknpuUnaryLayer {
public:
    RknpuNegLayer(LayerType ignore) : RknpuUnaryLayer(LAYER_NEG) {}
    ~RknpuNegLayer() {}

protected:
    Status Convert() {
        return RknpuUnaryLayer::UnaryConvert(rk::nn::OperatorType::NEG);
    }
};

REGISTER_RKNPU_LAYER(Neg, LAYER_NEG);

}  // namespace TNN_NS
