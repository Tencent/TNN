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

DECLARE_LAYER_INTERPRETER(Gather, LAYER_GATHER);

Status GatherLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int index, LayerParam** param) {
    auto layer_param = CreateLayerParam<GatherLayerParam>(param);
    GET_INT_1_OR_DEFAULT(layer_param->axis, 0);
    GET_INT_1_OR_DEFAULT(layer_param->data_in_resource, 0);
    GET_INT_1_OR_DEFAULT(layer_param->indices_in_resource, 1);
    return TNN_OK;
}

Status GatherLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    auto layer_res        = CreateLayerRes<GatherLayerResource>(resource);
    bool data_in_resource = deserializer.GetInt() == 1;
    if (data_in_resource) {
        GET_BUFFER_FOR_ATTR(layer_res, data, deserializer);
    }
    bool indices_in_resource = deserializer.GetInt() == 1;
    if (indices_in_resource) {
        GET_BUFFER_FOR_ATTR(layer_res, indices, deserializer);
    }
    
    if (data_in_resource && layer_res->data.GetDataType() == DATA_TYPE_INT8) {
        GET_BUFFER_FOR_ATTR(layer_res, scale_data, deserializer);
    }
    return TNN_OK;
}

Status GatherLayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
    auto layer_param = dynamic_cast<GatherLayerParam*>(param);
    if (layer_param == nullptr) {
        LOGE("invalid layer param to save\n");
        return Status(TNNERR_NULL_PARAM, "invalid layer param to save");
    }
    int data_in_resource    = layer_param->data_in_resource ? 1 : 0;
    int indices_in_resource = layer_param->indices_in_resource ? 1 : 0;
    output_stream << layer_param->axis << " ";
    output_stream << data_in_resource << " ";
    output_stream << indices_in_resource << " ";
    return TNN_OK;
}

Status GatherLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    auto layer_param = dynamic_cast<GatherLayerParam*>(param);
    auto layer_res   = dynamic_cast<GatherLayerResource*>(resource);
    if (layer_param == nullptr || layer_res == nullptr) {
        LOGE("Interpreter Gather: layer param or layer resource is null\n");
        return TNNERR_INVALID_MODEL;
    }
    if (layer_param->data_in_resource) {
        serializer.PutInt(1);
        serializer.PutRaw(layer_res->data);
    } else {
        serializer.PutInt(0);
    }
    if (layer_param->indices_in_resource) {
        serializer.PutInt(1);
        serializer.PutRaw(layer_res->indices);
    } else {
        serializer.PutInt(0);
    }
    
    if (layer_param->data_in_resource && layer_param->dynamic_range_quantized) {
        // now dynamic range quantization is to use symmetric quantization, only save scale
        serializer.PutRaw(layer_res->scale_data);
    }
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(Gather, LAYER_GATHER);

}  // namespace TNN_NS
