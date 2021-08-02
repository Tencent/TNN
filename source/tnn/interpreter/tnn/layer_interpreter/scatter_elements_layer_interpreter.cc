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
// COElementsITIONS OF ANY KIElements, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.
#include "abstract_layer_interpreter.h"

namespace TNN_NS {
DECLARE_LAYER_INTERPRETER(ScatterElements, LAYER_SCATTER_ELEMENTS);

Status ScatterElementsLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int start_index, LayerParam **param) {
    int index = start_index;
    auto p = CreateLayerParam<ScatterElementsLayerParam>(param);
    GET_INT_1(p->axis);

    return TNN_OK;
}

Status ScatterElementsLayerInterpreter::InterpretResource(Deserializer &deserializer, LayerResource **resource) {
    auto layer_resource = CreateLayerRes<ScatterElementsLayerResource>(resource);
    bool has_data    = deserializer.GetBool();
    if (has_data) {
        GET_BUFFER_FOR_ATTR(layer_resource, data, deserializer);
    }
    return TNN_OK;
}

Status ScatterElementsLayerInterpreter::SaveProto(std::ofstream &output_stream, LayerParam *param) {
    CAST_OR_RET_ERROR(layer_param, ScatterElementsLayerParam, "invalid scatter elements param to save", param);
    output_stream << layer_param->axis << " ";
    return TNN_OK;
}

Status ScatterElementsLayerInterpreter::SaveResource(Serializer &serializer, LayerParam *param, LayerResource *resource) {
    CAST_OR_RET_ERROR(layer_param, ScatterElementsLayerParam, "invalid layer param", param);
    auto layer_resource = dynamic_cast<ScatterElementsLayerResource *>(resource);
    if (!layer_resource) {
        return TNN_OK;
    }
    const auto &data_dims = layer_resource->data.GetBufferDims();
    bool has_data         = !data_dims.empty();
    if (has_data) {
        serializer.PutBool(true);
        serializer.PutRaw(layer_resource->data);
    } else {
        serializer.PutBool(false);
    }
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(ScatterElements, LAYER_SCATTER_ELEMENTS);

}  // namespace TNN_NS
