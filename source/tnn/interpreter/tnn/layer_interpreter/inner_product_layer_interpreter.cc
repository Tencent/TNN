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

DECLARE_LAYER_INTERPRETER(InnerProduct, LAYER_INNER_PRODUCT);

Status InnerProductLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int start_index, LayerParam** param) {
    InnerProductLayerParam* layer_param = new InnerProductLayerParam();
    *param                              = layer_param;
    int index                           = start_index;

    layer_param->num_output = atoi(layer_cfg_arr[index++].c_str());
    layer_param->has_bias   = atoi(layer_cfg_arr[index++].c_str());
    layer_param->transpose  = atoi(layer_cfg_arr[index++].c_str());
    layer_param->axis       = atoi(layer_cfg_arr[index++].c_str());

    return TNN_OK;
}

Status InnerProductLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    InnerProductLayerResource* layer_res = new InnerProductLayerResource();
    *resource                            = layer_res;

    std::string layer_name = deserializer.GetString();

    RawBuffer weights;
    deserializer.GetRaw(weights);
    layer_res->weight_handle = weights;

    RawBuffer bias;
    deserializer.GetRaw(bias);
    layer_res->bias_handle = bias;

    if (weights.GetDataType() == DATA_TYPE_INT8) {
        // Use the DataType of first_buffer to distinguish the old and new versions
        // old version: scale_handle(float)
        // new version: zero_point_handle(int8), scale_handle(float)
        RawBuffer first_buffer;
        deserializer.GetRaw(first_buffer);
        if (first_buffer.GetDataType() == DATA_TYPE_INT8) {
            layer_res->zero_point_handle = first_buffer;
            GET_BUFFER_FOR_ATTR(layer_res, scale_handle, deserializer);
        } else if (first_buffer.GetDataType() == DATA_TYPE_FLOAT) {
            layer_res->scale_handle = first_buffer;
            int total_byte_size     = first_buffer.GetDataCount() * sizeof(char);
            RawBuffer zero_point_buffer(total_byte_size);
            zero_point_buffer.SetDataType(DATA_TYPE_INT8);
            memset(zero_point_buffer.force_to<int8_t*>(), 0, total_byte_size);
            layer_res->zero_point_handle = zero_point_buffer;
        } else {
            LOGE("invalid quantized layer Resource\n");
            return -1;
        }
    }

    return TNN_OK;
}

Status InnerProductLayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
    InnerProductLayerParam* layer_param = dynamic_cast<InnerProductLayerParam*>(param);
    if (nullptr == layer_param) {
        LOGE("invalid layer param to save\n");
        return Status(TNNERR_NULL_PARAM, "invalid layer param to save");
    }

    output_stream << layer_param->num_output << " ";
    output_stream << layer_param->has_bias << " ";
    output_stream << layer_param->transpose << " ";
    output_stream << layer_param->axis << " ";

    return TNN_OK;
}

Status InnerProductLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    InnerProductLayerParam* layer_param = dynamic_cast<InnerProductLayerParam*>(param);
    if (nullptr == layer_param) {
        LOGE("invalid layer param to save\n");
        return Status(TNNERR_NULL_PARAM, "invalid layer param to save");
    }
    InnerProductLayerResource* layer_res = dynamic_cast<InnerProductLayerResource*>(resource);
    if (nullptr == layer_res) {
        LOGE("invalid layer res to save\n");
        return Status(TNNERR_NULL_PARAM, "invalid layer res to save");
    }

    serializer.PutString(layer_param->name);
    serializer.PutRaw(layer_res->weight_handle);
    serializer.PutRaw(layer_res->bias_handle);

    if (layer_param->quantized) {
        // put zero_point_handle in front of scale_handle to distinguish the old and new versions
        serializer.PutRaw(layer_res->zero_point_handle);
        serializer.PutRaw(layer_res->scale_handle);
    }

    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(InnerProduct, LAYER_INNER_PRODUCT);

}  // namespace TNN_NS
