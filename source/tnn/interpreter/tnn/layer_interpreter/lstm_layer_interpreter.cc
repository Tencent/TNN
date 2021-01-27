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

DECLARE_LAYER_INTERPRETER(LSTMONNX, LAYER_LSTMONNX);

Status LSTMONNXLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int index, LayerParam** param) {
    auto layer_param = CreateLayerParam<LSTMONNXLayerParam>(param);
    GET_FLOAT_1_OR_DEFAULT(layer_param->clip_threshold, 0);
    GET_INT_1_OR_DEFAULT(layer_param->hidden_size, 0);
    GET_INT_1_OR_DEFAULT(layer_param->direction, 0);
    return TNN_OK;
}

Status LSTMONNXLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    return TNN_OK;
}

Status LSTMONNXLayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
    auto layer_param = dynamic_cast<LSTMONNXLayerParam*>(param);
    if (layer_param == nullptr) {
        LOGE("invalid layer param to save\n");
        return Status(TNNERR_NULL_PARAM, "invalid layer param to save");
    }
    output_stream << layer_param->clip_threshold << " " << layer_param->hidden_size << " " << layer_param->direction << " ";

    return TNN_OK;
}

Status LSTMONNXLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(LSTMONNX, LAYER_LSTMONNX);

}  // namespace TNN_NS
