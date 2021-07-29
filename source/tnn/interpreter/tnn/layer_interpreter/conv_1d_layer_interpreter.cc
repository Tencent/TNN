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

DECLARE_LAYER_INTERPRETER(Conv1D, LAYER_CONVOLUTION_1D);

Status Conv1DLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int index, LayerParam** param) {
    auto p = CreateLayerParam<ConvLayerParam>(param);

    GET_INT_3(p->group, p->input_channel, p->output_channel);

    GET_INT_N_INTO_VEC_REVERSE(p->kernels, 1);

    GET_INT_N_INTO_VEC_REVERSE(p->strides, 1);

    int pad_w = 0, pad_h = 0;
    GET_INT_2(pad_h, pad_w);
    p->pads.push_back(pad_w);
    p->pads.push_back(pad_h);

    // bias
    GET_INT_2(p->bias, p->pad_type);

    GET_INT_N_INTO_VEC_REVERSE_DEFAULT(p->dialations, 1, 1);

    // activation
    GET_INT_1(p->activation_type);

    return TNN_OK;
}

Status Conv1DLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    auto conv_1d_res = CreateLayerRes<ConvLayerResource>(resource);

    // Todo. refactor later, for all data type
    conv_1d_res->filter_format = OIHW;

    std::string layer_name = deserializer.GetString();
    int has_bias           = deserializer.GetInt();

    GET_BUFFER_FOR_ATTR(conv_1d_res, filter_handle, deserializer);

    if (has_bias) {
        GET_BUFFER_FOR_ATTR(conv_1d_res, bias_handle, deserializer);
    }

    return TNN_OK;
}

Status Conv1DLayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
    CAST_OR_RET_ERROR(layer_param, ConvLayerParam, "invalid layer param to save", param);

    output_stream << layer_param->group << " ";
    output_stream << layer_param->input_channel << " ";
    output_stream << layer_param->output_channel << " ";

    ASSERT(layer_param->kernels.size() == 1);
    output_stream << layer_param->kernels[0] << " ";

    ASSERT(layer_param->strides.size() == 1);
    output_stream << layer_param->strides[0] << " ";

    ASSERT(layer_param->pads.size() == 2);
    output_stream << layer_param->pads[1] << " ";
    output_stream << layer_param->pads[0] << " ";

    output_stream << layer_param->bias << " ";
    output_stream << layer_param->pad_type << " ";

    ASSERT(layer_param->dialations.size() == 1);
    output_stream << layer_param->dialations[0] << " ";

    output_stream << layer_param->activation_type << " ";

    return TNN_OK;
}

Status Conv1DLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    CAST_OR_RET_ERROR(conv_1d_param, ConvLayerParam, "invalid layer param", param);
    CAST_OR_RET_ERROR(conv_1d_res, ConvLayerResource, "invalid layer res to save", resource);

    serializer.PutString(conv_1d_param->name);
    serializer.PutInt(conv_1d_param->bias);
    serializer.PutRaw(conv_1d_res->filter_handle);
    if (conv_1d_param->bias) {
        serializer.PutRaw(conv_1d_res->bias_handle);
    }

    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(Conv1D, LAYER_CONVOLUTION_1D);

}  // namespace TNN_NS
