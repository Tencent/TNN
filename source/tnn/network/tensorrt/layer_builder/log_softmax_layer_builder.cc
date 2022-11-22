// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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
#include "tnn/network/tensorrt/utils.h"

namespace TNN_NS {

DECLARE_TENSORRT_LAYER_BUILDER(LogSoftmax, LAYER_LOGSOFTMAX);

ILayer* LogSoftmaxTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto paramlist = dynamic_cast<LogSoftmaxLayerParam*>(param_);
    auto foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto input_tensor = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetTensor();
    ILayer* layer;
    auto tensor = GetInputITensors()[0];
    int size = tensor->getDimensions().nbDims;
    int axis = paramlist->axis;
    axis = axis < 0 ? axis + size : axis;

    ISoftMaxLayer* softmax_layer = network->addSoftMax(*input_tensor);
    if (softmax_layer != nullptr) {
        softmax_layer->setName((layer_name_+"_softmax").c_str());
        softmax_layer->setAxes(1 << axis);
        tensor = softmax_layer->getOutput(0);
    }
 
    ILayer* log_layer = network->addUnary(*tensor, UnaryOperation::kLOG);
    if (log_layer != nullptr) {
        log_layer->setName((layer_name_+"_log").c_str());
        layer = log_layer;
    }

    return layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(LogSoftmax, LAYER_LOGSOFTMAX);

}  //  namespace TNN_NS
