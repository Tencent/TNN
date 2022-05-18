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

DECLARE_LAYER_INTERPRETER(Range, LAYER_RANGE);

Status RangeLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int start_index, LayerParam** param) {
    auto layer_param = CreateLayerParam<RangeLayerParam>(param);
    int index = start_index;

    if (index+7>layer_cfg_arr.size()) {
        LOGE("Range TNN Interpreter: Range requires at least 7 layer param.");
        return Status(TNNERR_PARAM_ERR, "RangeLayerInterpreter param is invalid");
    }

    int dtype_value = std::atoi(layer_cfg_arr[index++].c_str());
    layer_param->data_type   = (DataType)dtype_value;
    layer_param->start.i     = std::atoi(layer_cfg_arr[index++].c_str());
    layer_param->limit.i     = std::atoi(layer_cfg_arr[index++].c_str());
    layer_param->delta.i     = std::atoi(layer_cfg_arr[index++].c_str());
    
    layer_param->start_index = std::atoi(layer_cfg_arr[index++].c_str());
    layer_param->limit_index = std::atoi(layer_cfg_arr[index++].c_str());
    layer_param->delta_index = std::atoi(layer_cfg_arr[index++].c_str());
 
    return TNN_OK;
}

Status RangeLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    return TNN_OK;
}

Status RangeLayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
    CAST_OR_RET_ERROR(layer_param, RangeLayerParam, "invalid layer param to save", param);
    
    output_stream << layer_param->data_type << " ";

    output_stream << layer_param->start.i << " ";
    output_stream << layer_param->limit.i << " ";
    output_stream << layer_param->delta.i << " ";

    output_stream << layer_param->start_index << " ";
    output_stream << layer_param->limit_index << " ";
    output_stream << layer_param->delta_index << " ";
    
    return TNN_OK;
}

Status RangeLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(Range, LAYER_RANGE);

}  // namespace TNN_NS
