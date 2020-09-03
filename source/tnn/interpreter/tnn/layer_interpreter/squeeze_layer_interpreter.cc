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

DECLARE_LAYER_INTERPRETER(Squeeze, LAYER_SQUEEZE);

Status SqueezeLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int index, LayerParam **param) {
    auto squeeze_param = CreateLayerParam<SqueezeLayerParam>(param);
    int size           = 0;
    GET_INT_1(size);
    for (int i = 0; i < size; ++i) {
        int axis = 0;
        GET_INT_1(axis);
        squeeze_param->axes.push_back(axis);
    }
    return TNN_OK;
}

Status SqueezeLayerInterpreter::InterpretResource(Deserializer &deserializer, LayerResource **Resource) {
    return TNN_OK;
}

Status SqueezeLayerInterpreter::SaveProto(std::ofstream &output_stream, LayerParam *param) {
    auto squeeze_param = dynamic_cast<SqueezeLayerParam *>(param);
    if (nullptr == squeeze_param) {
        LOGE("invalid layer param to save\n");
        return Status(TNNERR_NULL_PARAM, "invalid layer param to save");
    }
    output_stream << squeeze_param->axes.size() << " ";
    for (auto axis : squeeze_param->axes) {
        output_stream << axis << " ";
    }
    return TNN_OK;
}

Status SqueezeLayerInterpreter::SaveResource(Serializer &serializer, LayerParam *param, LayerResource *resource) {
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(Squeeze, LAYER_SQUEEZE);
}  // namespace TNN_NS