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

DECLARE_RKNPU_LAYER_WEIGHT(InnerProduct, LAYER_INNER_PRODUCT)

Status RknpuInnerProductLayer::Convert() {
    auto param    = dynamic_cast<InnerProductLayerParam *>(param_);
    auto resource = dynamic_cast<InnerProductLayerResource *>(resource_);
    CHECK_PARAM_NULL(param);
    if (!resource) {
        return Status(TNNERR_MODEL_ERR, "Error: InnerProductLayerResource is nil");
    }

    Status ret = TNN_OK;
    std::vector<std::shared_ptr<rk::nn::Tensor>> inputs;

    // input
    inputs.push_back(input_ops_[0]);

    // weight
    auto input_shape              = input_ops_[0]->GetDims();
    std::vector<int> weight_shape = {param->num_output, (int)input_shape[1], 1, 1};
    auto weight_const             = RknpuUtils::CreateRknnTensor(graph_, layer_name_ + "_weight", weight_shape,
                                                     resource->weight_handle.force_to<void *>(),
                                                     rk::nn::TensorRole::CONST, resource->weight_handle.GetDataType());
    inputs.push_back(weight_const);

    // bias
    int bias_count = resource->bias_handle.GetDataCount();
    if (param->has_bias) {
        std::vector<int> bias_shape = {1, bias_count, 1, 1};
        auto bias_const             = RknpuUtils::CreateRknnTensor(graph_, layer_name_ + "_bias", bias_shape,
                                                       resource->bias_handle.force_to<void *>(),
                                                       rk::nn::TensorRole::CONST, resource->bias_handle.GetDataType());
        inputs.push_back(bias_const);
    }

    // output
    ADD_OUTPUT_OP();

    rk::nn::FCAttr attr;
    attr.weights  = weight_shape[0];  // TODO
    attr.has_relu = false;

    graph_->AddOperator(rk::nn::OperatorType::FULLCONNECT, inputs, output_ops_, (void *)&attr);

    return ret;
}

REGISTER_RKNPU_LAYER(InnerProduct, LAYER_INNER_PRODUCT)

}  // namespace TNN_NS
