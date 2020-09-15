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

#include "tnn/network/tensorrt/layer_builder/tensorrt_plugin_layer_builder.h"

namespace TNN_NS {

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(Pow, LAYER_POWER);

bool PowTRTPluginLayerBuilder::supportsFormat(nvinfer1::DataType type, PluginFormat format) const {
    if (type == nvinfer1::DataType::kFLOAT) {
        return true;
    }
    return false;
}

ILayer* PowTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto paramlist = dynamic_cast<PowLayerParam *>(param_);
    if (paramlist->exponent == 1 && paramlist->scale == -1 && paramlist->shift == 0) {
        auto foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
        auto tensor = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetTensor();
        IUnaryLayer* layer = network->addUnary(*tensor, UnaryOperation::kNEG);
        if(layer != NULL) {
            layer->setName(layer_name_.c_str());
        }
        return layer;
    } else {
        return TensorRTPluginLayerBuilder::AddToNetwork(network);
    }
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(Pow, LAYER_POWER);

}  //  namespace TNN_NS