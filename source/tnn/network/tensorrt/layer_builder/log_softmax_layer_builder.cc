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

#include "tnn/network/tensorrt/layer_builder/tensorrt_layer_builder.h"

namespace TNN_NS {

DECLARE_TENSORRT_LAYER_BUILDER(LogSoftmax, LAYER_LOGSOFTMAX);

ILayer* LogSoftmaxTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto paramlist = dynamic_cast<LogSoftmaxLayerParam*>(param_);
    auto foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto input_tensor = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetTensor();
    ILayer* layer = nullptr;

    ISoftMaxLayer* softmax_layer = network->addSoftMax(*input_tensor);
    if (softmax_layer != nullptr) {
        const std::string softmax_suffix = "_softmax";
        softmax_layer->setName((layer_name_ + softmax_suffix).c_str());
        softmax_layer->setAxes(1 << paramlist->axis);
    } else {
        return layer;
    }

    IUnaryLayer* log_layer = network->addUnary(*softmax_layer->getOutput(0), UnaryOperation::kLOG);
    if (log_layer != nullptr) {
        const std::string log_suffix = "_log";
        log_layer->setName((layer_name_ + log_suffix).c_str());
        layer = log_layer;
    }
    return layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(LogSoftmax, LAYER_LOGSOFTMAX);

}  //  namespace TNN_NS
