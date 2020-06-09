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

#include "tnn/interpreter/tnn/layer_interpreter/abstract_layer_interpreter.h"

#include <stdlib.h>

namespace TNN_NS {

DECLARE_LAYER_INTERPRETER(LRN, LAYER_LRN);

Status LRNLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int start_index, LayerParam** param) {
    LRNLayerParam* layer_param = new LRNLayerParam();
    *param                     = layer_param;
    int index                  = start_index;

    // get min_size
    layer_param->alpha = static_cast<float>(atof(layer_cfg_arr[index++].c_str()));
    layer_param->beta  = static_cast<float>(atof(layer_cfg_arr[index++].c_str()));
    layer_param->bias  = static_cast<float>(atof(layer_cfg_arr[index++].c_str()));
    layer_param->size  = atoi(layer_cfg_arr[index++].c_str());

    return TNN_OK;
}

Status LRNLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    return TNN_OK;
}

Status LRNLayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
    auto layer_param = dynamic_cast<LRNLayerParam*>(param);
    if (nullptr == layer_param) {
        LOGE("invalid layer param to save\n");
        return Status(TNNERR_NULL_PARAM, "invalid layer param to save");
    }
    output_stream << layer_param->alpha << " ";
    output_stream << layer_param->beta << " ";
    output_stream << layer_param->bias << " ";
    output_stream << layer_param->size << " ";

    return TNN_OK;
}

Status LRNLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(LRN, LAYER_LRN);

}  // namespace TNN_NS
