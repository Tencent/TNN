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

#include <cmath>
#include <memory>

#include <ngraph/node.hpp>
#include <ngraph/op/op.hpp>

#include "tnn/layer/base_layer.h"
#include "tnn/device/openvino/layer_builder/openvino_layer_builder.h"
#include "tnn/device/openvino/common/foreign_blob.h"
#include "tnn/device/openvino/common/foreign_tensor.h"
#include "tnn/device/openvino/openvino_types.h"

namespace TNN_NS {

DECLARE_OPENVINO_LAYER_BUILDER(Conv, LAYER_CONVOLUTION);

Status ConvOVLayerBuilder::Build() {
    if (GetInputNodes().size() <=0) {
        LOGE("Error: 0 input nodes\n");
        return TNNERR_INIT_LAYER;
    }
    
    auto in_node = GetInputNodes()[0];
    
    // build the conv layer and generates a new out_node. 
    // here simply asign out_node = in_node for code compiling.
    auto out_node = in_node;

    if (GetOutputTensors().size() <=0) {
        LOGE("Error: 0 output tensors\n");
        return TNNERR_INIT_LAYER;
    }

    std::dynamic_pointer_cast<OpenvinoTensor>(GetOutputTensors()[0])->SetNode(out_node);

    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(Conv, LAYER_CONVOLUTION);

}  // namespace TNN_NS
