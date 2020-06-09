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

#include <map>
#include <vector>

#include "tnn/interpreter/ncnn/layer_interpreter/abstract_layer_interpreter.h"
#include "tnn/interpreter/ncnn/ncnn_layer_type.h"
#include "tnn/interpreter/ncnn/ncnn_param_utils.h"

namespace TNN_NS {

namespace ncnn {

    DECLARE_LAYER_INTERPRETER(Permute);

    REGISTER_LAYER_INTERPRETER(Permute, Permute);

    /* ncnn permute from c h w to :
    order_type:
        0 = c h w
        1 = c w h
        2 = h c w
        3 = h w c
        4 = w c h
        5 = w h c
    */

    std::map<int, std::vector<int>> order_type_map = {
        {0, {0, 1, 2, 3}}, {1, {0, 1, 3, 2}}, {2, {0, 2, 1, 3}},
        {3, {0, 2, 3, 1}}, {4, {0, 3, 1, 2}}, {5, {0, 3, 2, 1}},
    };

    Status PermuteLayerInterpreter::InterpretProto(std::string type_name, str_dict param_dict, LayerType& type,
                                                   LayerParam** param) {
        PermuteLayerParam* layer_param = new PermuteLayerParam();
        *param                         = layer_param;

        type = ConvertNCNNLayerType(type_name);

        auto& p = param_dict;

        int order_type      = GetInt(p, 0, 0);
        layer_param->orders = order_type_map[order_type];

        return TNN_OK;
    }

    Status PermuteLayerInterpreter::InterpretResource(Deserializer& deserializer, std::shared_ptr<LayerInfo> info,
                                                      LayerResource** resource) {
        return TNN_OK;
    }

}  // namespace ncnn

}  // namespace TNN_NS
