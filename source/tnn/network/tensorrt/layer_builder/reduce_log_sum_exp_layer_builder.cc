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

#include "tnn/network/tensorrt/layer_builder/reduce_log_sum_exp_layer_builder.h"

namespace TNN_NS {

ReduceLogSumExpTRTLayerBuilder::ReduceLogSumExpTRTLayerBuilder(LayerType ignore) : ReduceTRTLayerBuilder(ignore) {
}

ILayer* ReduceLogSumExpTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto paramlist = dynamic_cast<ReduceLayerParam*>(param_);
    auto input_tensors = GetInputITensors();
    ILayer* exp_layer    = network->addUnary(*input_tensors[0], UnaryOperation::kEXP);
    ILayer* reduce_layer = network->addReduce(*exp_layer->getOutput(0), ReduceOperation::kSUM,
                                                GetReduceAxis(), paramlist->keep_dims == 1);
    ILayer* log_layer    = network->addUnary(*reduce_layer->getOutput(0), UnaryOperation::kLOG);
    if (log_layer != nullptr) {
        log_layer->setName(layer_name_.c_str());
    }
    
    return log_layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(ReduceLogSumExp, LAYER_REDUCE_LOG_SUM_EXP);

}  //  namespace TNN_NS

