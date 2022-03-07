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

DECLARE_LAYER_INTERPRETER(MatMul, LAYER_MATMUL);

Status MatMulLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int index, LayerParam** param) {
    auto layer_param = new MatMulLayerParam();
    *param            = layer_param;
    if (index < layer_cfg_arr.size()) {
       layer_param->weight_position = atoi(layer_cfg_arr[index++].c_str());
    }
    return TNN_OK;
}

Status MatMulLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    auto layer_res = CreateLayerRes<MatMulLayerResource>(resource);
    RawBuffer buf;
    deserializer.GetRaw(buf);
    layer_res->weight = buf;

    if (layer_res->weight.GetDataType() == DATA_TYPE_INT8) {
        RawBuffer scale;
        deserializer.GetRaw(scale);
        layer_res->scale_handle = scale;
    }

    return TNN_OK;
}

Status MatMulLayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
    auto layer_param = dynamic_cast<MatMulLayerParam*>(param);
    if (nullptr == layer_param) {
        return Status(TNNERR_NULL_PARAM, "invalid layer param to save");
    }
    output_stream << layer_param->weight_position << " ";
    return TNN_OK;
}

Status MatMulLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    CAST_OR_RET_ERROR(layer_res, MatMulLayerResource, "invalid layer res to save", resource);
    serializer.PutRaw(layer_res->weight);
    if (param->dynamic_range_quantized) {
        serializer.PutRaw(layer_res->scale_handle);
    }
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(MatMul, LAYER_MATMUL);

}  // namespace TNN_NS
