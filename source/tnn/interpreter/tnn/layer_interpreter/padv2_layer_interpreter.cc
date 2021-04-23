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

#include <limits.h>
#include <stdlib.h>

#include "tnn/interpreter/tnn/layer_interpreter/abstract_layer_interpreter.h"

namespace TNN_NS {

DECLARE_LAYER_INTERPRETER(PadV2, LAYER_PADV2);

Status PadV2LayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int start_index, LayerParam** param) {
    auto layer_param = new PadLayerParam();
    *param           = layer_param;
    int index        = start_index;
    
    int dim_size = 0;
    if (index < layer_cfg_arr.size()) {
        dim_size = atoi(layer_cfg_arr[index++].c_str());
    }
    
    DimsVector pads;
    for (int i=0; i<2*dim_size; i++) {
        pads.push_back(atoi(layer_cfg_arr[index++].c_str()));
    }
    layer_param->pads = pads;
    
    if (index < layer_cfg_arr.size()) {
        layer_param->type = atoi(layer_cfg_arr[index++].c_str());
    }
    if (index < layer_cfg_arr.size()) {
        layer_param->value = atof(layer_cfg_arr[index++].c_str());
    }
    return TNN_OK;
}

Status PadV2LayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    return TNN_OK;
}

Status PadV2LayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
    auto layer_param = dynamic_cast<PadLayerParam*>(param);
    if (nullptr == layer_param) {
        LOGE("invalid layer param to save\n");
        return Status(TNNERR_NULL_PARAM, "invalid layer param to save");
    }
    
    const auto pads = layer_param->pads;
    int dim_size = (int)pads.size()/2;
    output_stream << dim_size << " ";
    for (int i=0; i<pads.size(); i++) {
        output_stream << pads[i] << " ";
    }
    
    output_stream << layer_param->type << " "<< layer_param->value << " ";
    return TNN_OK;
}

Status PadV2LayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(PadV2, LAYER_PADV2);

}  // namespace TNN_NS
