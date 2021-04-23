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

DECLARE_LAYER_INTERPRETER(Histogram, LAYER_HISTOGRAM);

Status HistogramLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int index, LayerParam **param) {
    auto layer_param = CreateLayerParam<HistogramLayerParam>(param);
    GET_INT_1_OR_DEFAULT(layer_param->depth, 0);
    return TNN_OK;
}

Status HistogramLayerInterpreter::InterpretResource(Deserializer &deserializer, LayerResource **Resource) {
    return TNN_OK;
}

Status HistogramLayerInterpreter::SaveProto(std::ofstream &output_stream, LayerParam *param) {
    auto layer_param = dynamic_cast<HistogramLayerParam *>(param);
    if (nullptr == layer_param) {
        LOGE("invalid layer param to save\n");
        return Status(TNNERR_NULL_PARAM, "invalid layer param to save");
    }
    output_stream << layer_param->depth << " ";
    return TNN_OK;
}

Status HistogramLayerInterpreter::SaveResource(Serializer &serializer, LayerParam *param, LayerResource *resource) {
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(Histogram, LAYER_HISTOGRAM);
}  // namespace TNN_NS
