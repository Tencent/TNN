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

#include "rknpu_base_layer.h"
#include "rknpu_utils.h"

namespace TNN_NS {

DECLARE_RKNPU_LAYER(Pool, LAYER_POOLING)

Status RknpuPoolLayer::Convert() {
    // parameter and weight of the pooling layer
    auto param = dynamic_cast<PoolingLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    rk::nn::PadType rk_pad_type = rk::nn::PadType::AUTO;
    Status ret                  = RknpuUtils::GetPadType(rk_pad_type, param->pad_type);
    if (ret != TNN_OK)
        return ret;

    std::vector<std::shared_ptr<rk::nn::Tensor>> inputs;

    // input
    inputs.push_back(input_ops_[0]);

    // output
    ADD_OUTPUT_OP();

    rk::nn::PoolAttr attr;
    attr.ksize[0]       = param->kernels[0];
    attr.ksize[1]       = param->kernels[1];
    attr.stride[0]      = param->strides[0];
    attr.stride[1]      = param->strides[1];
    attr.pad[0]         = param->pads[0];
    attr.pad[1]         = param->pads[1];
    attr.pad[2]         = param->pads[2];
    attr.pad[3]         = param->pads[3];
    attr.pad_type       = rk_pad_type;
    attr.pool_type      = (0 == param->pool_type) ? rk::nn::PoolType::POOLING_MAX : rk::nn::PoolType::POOLING_AVG;
    attr.round_type     = (1 == param->ceil_mode) ? rk::nn::RoundType::ROUND_CEIL : rk::nn::RoundType::ROUND_FLOOR;
    attr.global_pooling = (attr.ksize[0] == -1 && attr.ksize[1] == -1);

    graph_->AddOperator(rk::nn::OperatorType::POOL, inputs, output_ops_, (void *)&attr);
    return ret;
}

REGISTER_RKNPU_LAYER(Pool, LAYER_POOLING)

}  // namespace TNN_NS
