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

namespace TNN_NS {

DECLARE_LAYER_INTERPRETER(InstanceNorm, LAYER_INST_BATCH_NORM);

Status InstanceNormLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int index, LayerParam** param) {
    auto p = CreateLayerParam<InstanceNormLayerParam>(param);
    GET_INT_1_OR_DEFAULT(p->channels, 0);
    GET_FLOAT_1_OR_DEFAULT(p->eps, 1e-5f);
    return TNN_OK;
}

Status InstanceNormLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    auto instnorm_res = CreateLayerRes<InstanceNormLayerResource>(resource);

    GET_BUFFER_FOR_ATTR(instnorm_res, scale_handle, deserializer);
    GET_BUFFER_FOR_ATTR(instnorm_res, bias_handle, deserializer);

    if (instnorm_res->bias_handle.GetBytesSize() == 0) {
        size_t scal_byte_size     = instnorm_res->scale_handle.GetBytesSize();
        instnorm_res->bias_handle = RawBuffer(scal_byte_size);
    }

    return TNN_OK;
}

Status InstanceNormLayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
    CAST_OR_RET_ERROR(layer_param, InstanceNormLayerParam, "invalid group norm layer param to save", param);
    output_stream << layer_param->channels << " ";
    output_stream << layer_param->eps << " ";
    return TNN_OK;
}

Status InstanceNormLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    CAST_OR_RET_ERROR(instnorm_res, InstanceNormLayerResource, "invalid layer res to save", resource);
    serializer.PutRaw(instnorm_res->scale_handle);
    serializer.PutRaw(instnorm_res->bias_handle);
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(InstanceNorm, LAYER_INST_BATCH_NORM);

}  // namespace TNN_NS
