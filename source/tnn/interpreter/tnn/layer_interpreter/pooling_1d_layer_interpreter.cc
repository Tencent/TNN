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

DECLARE_LAYER_INTERPRETER(Pooling1D, LAYER_POOLING_1D);

Status Pooling1DLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int index, LayerParam** param) {
    auto p = CreateLayerParam<PoolingLayerParam>(param);

    GET_INT_1(p->pool_type);

    GET_INT_N_INTO_VEC_REVERSE(p->kernels, 1);
    p->kernels_params = p->kernels;

    GET_INT_N_INTO_VEC_REVERSE(p->strides, 1);

    int pad = 0;
    GET_INT_1(pad);
    p->pads.push_back(pad);
    p->pads.push_back(pad);

    GET_INT_N_INTO_VEC_REVERSE_DEFAULT(p->kernel_indexs, 2, -1);

    GET_INT_2(p->pad_type, p->ceil_mode);

    return TNN_OK;
}

Status Pooling1DLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    return TNN_OK;
}

Status Pooling1DLayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
    CAST_OR_RET_ERROR(layer_param, PoolingLayerParam, "invalid layer param to save", param);

    output_stream << layer_param->pool_type << " ";

    ASSERT(layer_param->kernels_params.size() == 1);
    output_stream << layer_param->kernels_params[0] << " ";

    ASSERT(layer_param->strides.size() == 1);
    output_stream << layer_param->strides[0] << " ";

    ASSERT(layer_param->pads.size() == 2);
    output_stream << layer_param->pads[0] << " ";

    ASSERT(layer_param->kernel_indexs.size() == 2);
    output_stream << layer_param->kernel_indexs[1] << " ";
    output_stream << layer_param->kernel_indexs[0] << " ";

    output_stream << layer_param->pad_type << " ";
    output_stream << layer_param->ceil_mode << " ";

    return TNN_OK;
}

Status Pooling1DLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(Pooling1D, LAYER_POOLING_1D);

}  // namespace TNN_NS
