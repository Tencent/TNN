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

#include <stdlib.h>

#include "tnn/interpreter/tnn/layer_interpreter/abstract_layer_interpreter.h"

namespace TNN_NS {

DECLARE_LAYER_INTERPRETER(Reformat, LAYER_MAXIMUM);

DataType GetDataType(int type_value) {
    switch (type_value) {
        case DATA_TYPE_FLOAT:
            return DATA_TYPE_FLOAT;
        case DATA_TYPE_HALF:
            return DATA_TYPE_HALF;
        case DATA_TYPE_INT8:
            return DATA_TYPE_INT8;
        case DATA_TYPE_INT32:
            return DATA_TYPE_INT32;
        case DATA_TYPE_BFP16:
            return DATA_TYPE_BFP16;
        default:
            LOGE("Interpreter: do not support reformat src type");
            return DATA_TYPE_FLOAT;
    }
}

Status ReformatLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int start_index, LayerParam** param) {
    auto layer_param   = CreateLayerParam<ReformatLayerParam>(param);
    int index          = start_index;
    int src_type_value = 0;
    if (index < layer_cfg_arr.size()) {
        src_type_value = atoi(layer_cfg_arr[index++].c_str());
    }
    layer_param->src_type = GetDataType(src_type_value);
    int dst_type_value    = 0;
    if (index < layer_cfg_arr.size()) {
        dst_type_value = atoi(layer_cfg_arr[index++].c_str());
    }
    layer_param->dst_type = GetDataType(dst_type_value);
    return TNN_OK;
}

Status ReformatLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    return TNN_OK;
}

Status ReformatLayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
    auto layer_param = dynamic_cast<ReformatLayerParam*>(param);
    output_stream << layer_param->src_type << " ";
    output_stream << layer_param->dst_type << " ";
    return TNN_OK;
}

Status ReformatLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(Reformat, LAYER_REFORMAT);

}  // namespace TNN_NS
