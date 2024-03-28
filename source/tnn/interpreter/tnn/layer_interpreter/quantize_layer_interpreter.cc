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
DECLARE_LAYER_INTERPRETER(Quantize, LAYER_QUANTIZE);

Status QuantizeLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int start_index, LayerParam **param) {
    auto *layer_param = new QuantizeLayerParam();
    *param            = layer_param;
    int index         = start_index;

    GET_INT_1_OR_DEFAULT(layer_param->axis, 0);

    return TNN_OK;
}

Status QuantizeLayerInterpreter::InterpretResource(Deserializer &deserializer, LayerResource **resource) {
    auto layer_resource = CreateLayerRes<QuantizeLayerResource>(resource);

    GET_BUFFER_FOR_ATTR(layer_resource, scale_handle, deserializer);

    return TNN_OK;
}

Status QuantizeLayerInterpreter::SaveProto(std::ostream &output_stream, LayerParam *param) {
    auto *layer_param = static_cast<QuantizeLayerParam *>(param);
    if (nullptr == layer_param) {
        LOGE("invalid layer param to save\n");
        return Status(TNNERR_NULL_PARAM, "invalid layer param to save");
    }

    output_stream << layer_param->axis << " ";

    return TNN_OK;
}

Status QuantizeLayerInterpreter::SaveResource(Serializer &serializer, LayerParam *param, LayerResource *resource) {
    auto layer_resource = dynamic_cast<QuantizeLayerResource *>(resource);
    if (nullptr == layer_resource) {
        LOGE("invalid layer resourve to save\n");
        return Status(TNNERR_NULL_PARAM, "invalid layer resource to save");
    }

    serializer.PutRaw(layer_resource->scale_handle);

    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(Quantize, LAYER_QUANTIZE);
REGISTER_LAYER_INTERPRETER(Quantize, LAYER_DEQUANTIZE);

}  // namespace TNN_NS
