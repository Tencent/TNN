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

#include "tnn/network/openvino/custom_layer/custom_reorg.h"
#include "tnn/network/openvino/layer_builder/openvino_layer_builder.h"
#include "tnn/network/openvino/utils.h"

namespace TNN_NS {

DECLARE_OPENVINO_LAYER_BUILDER(Reorg, LAYER_REORG);

Status ReorgOVLayerBuilder::Build() {
    
    auto paramlist = dynamic_cast<ReorgLayerParam*>(param_);

    if (GetInputNodes().size() <=0) {
        LOGE("Error: 0 input nodes\n");
        return TNNERR_INIT_LAYER;
    }

    // custom x86 reorg layer
    ADD_CUSTOM_NODE(Reorg, paramlist->name);

    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(Reorg, LAYER_REORG);
REGISTER_CUSTOM_TYPE(LAYER_REORG);

}
