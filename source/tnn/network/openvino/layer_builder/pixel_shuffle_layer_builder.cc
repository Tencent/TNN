// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "tnn/network/openvino/layer_builder/openvino_layer_builder.h"

namespace TNN_NS {

DECLARE_OPENVINO_LAYER_BUILDER(PixelShuffle, LAYER_PIXEL_SHUFFLE);

Status PixelShuffleOVLayerBuilder::Build() {
    
    auto paramlist = dynamic_cast<PixelShuffleLayerParam*>(param_);

    auto inputNodes = GetInputNodes();
    if (inputNodes.size() != 1) {
        LOGE("Error: pixel shuffle layer input counts mismatch\n");
        return TNNERR_INIT_LAYER;
    }

    auto mode = ngraph::op::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST;

    auto pixelShuffleNode = std::make_shared<ngraph::op::DepthToSpace>(inputNodes[0], mode, paramlist->upscale_factor);
    pixelShuffleNode->set_friendly_name(paramlist->name);

    ngraph::NodeVector outputNodes = {pixelShuffleNode};
    SetOutputTensors(outputNodes);

    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(PixelShuffle, LAYER_PIXEL_SHUFFLE);

}
