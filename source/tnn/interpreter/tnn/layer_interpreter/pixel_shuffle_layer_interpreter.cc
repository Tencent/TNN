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

#include "tnn/core/layer_type.h"
#include "tnn/interpreter/tnn/layer_interpreter/abstract_layer_interpreter.h"
namespace TNN_NS {

DECLARE_LAYER_INTERPRETER(PixelShuffle, LAYER_PIXEL_SHUFFLE);

Status PixelShuffleLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int start_index, LayerParam **param) {
    auto layer_param            = new PixelShuffleLayerParam();
    *param                      = layer_param;
    int index                   = start_index;
    int upscale_factor          = atoi(layer_cfg_arr[index++].c_str());
    layer_param->upscale_factor = upscale_factor;
    return TNN_OK;
}

Status PixelShuffleLayerInterpreter::InterpretResource(Deserializer &deserializer, LayerResource **Resource) {
    return TNN_OK;
}

Status PixelShuffleLayerInterpreter::SaveProto(std::ofstream &output_stream, LayerParam *param) {
    auto layer_param = dynamic_cast<PixelShuffleLayerParam *>(param);
    CHECK_PARAM_NULL(layer_param);
    output_stream << layer_param->upscale_factor << " ";
    return TNN_OK;
}

Status PixelShuffleLayerInterpreter::SaveResource(Serializer &serializer, LayerParam *param, LayerResource *resource) {
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(PixelShuffle, LAYER_PIXEL_SHUFFLE);
}  // namespace TNN_NS