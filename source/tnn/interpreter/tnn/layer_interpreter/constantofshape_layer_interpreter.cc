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

#include "abstract_layer_interpreter.h"

namespace TNN_NS {

DECLARE_LAYER_INTERPRETER(ConstantOfShape, LAYER_CONSTANT_OF_SHAPE);

Status ConstantOfShapeLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int index, LayerParam** param) {
    return TNN_OK;
}

Status ConstantOfShapeLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    auto layer_res = CreateLayerRes<ConstantOfShapeLayerResource>(resource);
    GET_BUFFER_FOR_ATTR(layer_res, value, deserializer);

    return TNN_OK;
}

Status ConstantOfShapeLayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
    return TNN_OK;
}

Status ConstantOfShapeLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    auto layer_res   = dynamic_cast<ConstantOfShapeLayerResource*>(resource);
    RETURN_VALUE_ON_NEQ(!layer_res, false,
                        Status(TNNERR_INVALID_MODEL, "ConstantOfShape value is empty\n"));
    
    serializer.PutRaw(layer_res->value);
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(ConstantOfShape, LAYER_CONSTANT_OF_SHAPE);

}  // namespace TNN_NS
