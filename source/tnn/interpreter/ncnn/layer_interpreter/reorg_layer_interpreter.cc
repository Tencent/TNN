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

#include <cmath>

#include "tnn/interpreter/ncnn/layer_interpreter/abstract_layer_interpreter.h"
#include "tnn/interpreter/ncnn/ncnn_layer_type.h"
#include "tnn/interpreter/ncnn/ncnn_param_utils.h"

namespace TNN_NS {

namespace ncnn {

    DECLARE_LAYER_INTERPRETER(Reorg);

    REGISTER_LAYER_INTERPRETER(Reorg, Reorg);

    Status ReorgLayerInterpreter::InterpretProto(std::string type_name, str_dict param_dict, LayerType& type,
                                                     LayerParam** param) {
        ReorgLayerParam* layer_param = new ReorgLayerParam();
        *param                           = layer_param;

        type = ConvertNCNNLayerType(type_name);

        auto& p               = param_dict;

        layer_param->stride = GetInt(p, 0, 0);
        layer_param->forward = false;


        return TNN_OK;
    }

    Status ReorgLayerInterpreter::InterpretResource(Deserializer& deserializer, std::shared_ptr<LayerInfo> info,
                                                        LayerResource** resource) {

        return TNN_OK;
    }

}  // namespace ncnn

}  // namespace TNN_NS
