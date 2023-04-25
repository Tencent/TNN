// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

DECLARE_LAYER_INTERPRETER(Fused, LAYER_FUSED);

Status FusedLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int index, LayerParam** param) {
    auto layer_param = CreateLayerParam<FusedLayerParam>(param);
    int fused_type = -1;
    GET_INT_1_OR_DEFAULT(fused_type, -1);
    layer_param->type = static_cast<FusionType>(fused_type);
    GET_INT_1_OR_DEFAULT(layer_param->bert_mha_hidden_size, -1);
    GET_INT_1_OR_DEFAULT(layer_param->bert_mha_num_heads, -1);

    GET_INT_1_OR_DEFAULT(layer_param->layer_norm_param.reduce_dims_size, 0);
    GET_FLOAT_1_OR_DEFAULT(layer_param->layer_norm_param.eps, 1e-5f);

    int ffn_activation = 0;
    GET_INT_1_OR_DEFAULT(ffn_activation, 0);
    layer_param->ffn_activation = static_cast<ActivationType>(ffn_activation);
    GET_INT_1_OR_DEFAULT(layer_param->ffn_inter_size, 0);

    GET_INT_1_OR_DEFAULT(layer_param->attention_head_num, -1);
    GET_INT_1_OR_DEFAULT(layer_param->attention_size_per_head, -1);
    GET_FLOAT_1_OR_DEFAULT(layer_param->attention_q_scaling, 1.0f);

    return TNN_OK;
}

Status FusedLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    auto layer_res   = CreateLayerRes<FusedLayerResource>(resource);

    RawBuffer ffn_matmul_in;
    deserializer.GetRaw(ffn_matmul_in);
    layer_res->ffn_matmul_in.weight = ffn_matmul_in;

    RawBuffer ffn_matmul_out;
    deserializer.GetRaw(ffn_matmul_out);
    layer_res->ffn_matmul_out.weight = ffn_matmul_out;

    RawBuffer ffn_bias;
    deserializer.GetRaw(ffn_bias);
    layer_res->ffn_bias.element_handle = ffn_bias;

    RawBuffer attention_q_mm;
    deserializer.GetRaw(attention_q_mm);
    layer_res->attention_q_mm.weight = attention_q_mm;

    RawBuffer attention_k_mm;
    deserializer.GetRaw(attention_k_mm);
    layer_res->attention_k_mm.weight = attention_k_mm;

    RawBuffer attention_v_mm;
    deserializer.GetRaw(attention_v_mm);
    layer_res->attention_v_mm.weight = attention_v_mm;

    RawBuffer attention_o_mm;
    deserializer.GetRaw(attention_o_mm);
    layer_res->attention_o_mm.weight = attention_o_mm;

    RawBuffer attention_q_bias;
    deserializer.GetRaw(attention_q_bias);
    layer_res->attention_q_bias.element_handle = attention_q_bias;

    RawBuffer attention_k_bias;
    deserializer.GetRaw(attention_k_bias);
    layer_res->attention_k_bias.element_handle = attention_k_bias;

    RawBuffer attention_v_bias;
    deserializer.GetRaw(attention_v_bias);
    layer_res->attention_v_bias.element_handle = attention_v_bias;

    return TNN_OK;
}

Status FusedLayerInterpreter::SaveProto(std::ostream& output_stream, LayerParam* param) {
    CAST_OR_RET_ERROR(layer_param, FusedLayerParam, "invalid layer param to save", param);
    output_stream << int(layer_param->type) << " ";
    output_stream << int(layer_param->bert_mha_hidden_size) << " ";
    output_stream << int(layer_param->bert_mha_num_heads) << " ";

    output_stream << layer_param->layer_norm_param.reduce_dims_size << " ";
    output_stream << layer_param->layer_norm_param.eps << " ";

    output_stream << int(layer_param->ffn_activation) << " ";
    output_stream << layer_param->ffn_inter_size << " ";

    output_stream << layer_param->attention_head_num << " ";
    output_stream << layer_param->attention_size_per_head << " ";
    output_stream << layer_param->attention_q_scaling << " ";

    return TNN_OK;
}

Status FusedLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    CAST_OR_RET_ERROR(layer_param, FusedLayerParam, "invalid layer param when save resource", param);
    FusedLayerResource empty_res;
    FusedLayerResource *layer_res = &empty_res;
    if (layer_param->type == FusionType_FFN ||
        layer_param->type == FusionType_Attention) {
        CAST_OR_RET_ERROR(fused_layer_res, FusedLayerResource, "invalid layer res to save", resource);
        layer_res = fused_layer_res;
    }

    serializer.PutRaw(layer_res->ffn_matmul_in.weight);
    serializer.PutRaw(layer_res->ffn_matmul_out.weight);
    serializer.PutRaw(layer_res->ffn_bias.element_handle);
    serializer.PutRaw(layer_res->attention_q_mm.weight);
    serializer.PutRaw(layer_res->attention_k_mm.weight);
    serializer.PutRaw(layer_res->attention_v_mm.weight);
    serializer.PutRaw(layer_res->attention_o_mm.weight);
    serializer.PutRaw(layer_res->attention_q_bias.element_handle);
    serializer.PutRaw(layer_res->attention_k_bias.element_handle);
    serializer.PutRaw(layer_res->attention_v_bias.element_handle);
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(Fused, LAYER_FUSED);

}  // namespace TNN_NS
