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
#include "tnn/interpreter/tnn/layer_interpreter/layer_interpreter_macro.h"

#include <stdlib.h>

namespace TNN_NS {

DECLARE_LAYER_INTERPRETER(Conv, LAYER_CONVOLUTION);

Status ConvLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int start_index, LayerParam** param) {
    int index = start_index;

    auto p = CreateLayerParam<ConvLayerParam>(param);

    GET_INT_3(p->group, p->input_channel, p->output_channel);

    // kernels h,w -> w,h
    GET_INT_2_INTO_VEC_REVERSE(p->kernels);

    // strides h,w -> w,h
    GET_INT_2_INTO_VEC_REVERSE(p->strides);

    // pads
    int pad_h = 0, pad_w = 0;
    GET_INT_2(pad_h, pad_w);
    p->pads.push_back(pad_w);
    p->pads.push_back(pad_w);
    p->pads.push_back(pad_h);
    p->pads.push_back(pad_h);

    // bias
    GET_INT_2(p->bias, p->pad_type);

    // dailations
    GET_INT_2_INTO_VEC_REVERSE_DEFAULT(p->dialations, 1);

    // activation
    GET_INT_1(p->activation_type);

    return TNN_OK;
}

Status ConvLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    auto layer_res = CreateLayerRes<ConvLayerResource>(resource);

    // Todo. refactor later, for all data type
    layer_res->filter_format = OIHW;

    std::string layer_name = deserializer.GetString();
    int has_bias           = deserializer.GetInt();

    GET_BUFFER_FOR_ATTR(layer_res, filter_handle, deserializer);

    if (has_bias) {
        GET_BUFFER_FOR_ATTR(layer_res, bias_handle, deserializer);
    }

    if (layer_res->filter_handle.GetDataType() == DATA_TYPE_INT8) {
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

Status ConvLayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
    CAST_OR_RET_ERROR(layer_param, ConvLayerParam, "invalid layer param to save", param);

    output_stream << layer_param->group << " ";
    output_stream << layer_param->input_channel << " ";
    output_stream << layer_param->output_channel << " ";

    ASSERT(layer_param->kernels.size() == 2);
    output_stream << layer_param->kernels[1] << " ";
    output_stream << layer_param->kernels[0] << " ";

    ASSERT(layer_param->strides.size() == 2);
    output_stream << layer_param->strides[1] << " ";
    output_stream << layer_param->strides[0] << " ";

    ASSERT(layer_param->pads.size() == 4);
    output_stream << layer_param->pads[2] << " ";
    output_stream << layer_param->pads[0] << " ";

    output_stream << layer_param->bias << " ";
    output_stream << layer_param->pad_type << " ";

    ASSERT(layer_param->dialations.size() == 2);
    output_stream << layer_param->dialations[1] << " ";
    output_stream << layer_param->dialations[0] << " ";

    output_stream << layer_param->activation_type << " ";

    return TNN_OK;
}

Status ConvLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    CAST_OR_RET_ERROR(layer_param, ConvLayerParam, "invalid layer param", param);
    CAST_OR_RET_ERROR(layer_res, ConvLayerResource, "invalid layer res to save", resource);

    serializer.PutString(layer_param->name);
    serializer.PutInt(layer_param->bias);
    serializer.PutRaw(layer_res->filter_handle);
    if (layer_param->bias) {
        serializer.PutRaw(layer_res->bias_handle);
    }
    if (layer_param->quantized) {
        // put zero_point_handle in front of scale_handle to distinguish the old and new versions
        serializer.PutRaw(layer_res->zero_point_handle);
        serializer.PutRaw(layer_res->scale_handle);
    }
    if (layer_param->dynamic_range_quantized) {
        // now dynamic range quantization is to use symmetric quantization, only save scale
        serializer.PutRaw(layer_res->scale_handle);
    }

    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(Conv, LAYER_CONVOLUTION);
REGISTER_LAYER_INTERPRETER(Conv, LAYER_DECONVOLUTION);

}  // namespace TNN_NS
