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

#include "tnn/interpreter/ncnn/layer_interpreter/abstract_layer_interpreter.h"
#include "tnn/interpreter/ncnn/ncnn_layer_type.h"
#include "tnn/interpreter/ncnn/ncnn_param_utils.h"

namespace TNN_NS {

namespace ncnn {

    DECLARE_LAYER_INTERPRETER(Pad);

    REGISTER_LAYER_INTERPRETER(Pad, Padding);

    Status PadLayerInterpreter::InterpretProto(std::string type_name, str_dict param_dict, LayerType& type,
                                               LayerParam** param) {
        type = ConvertNCNNLayerType(type_name);

        PadLayerParam* layer_param = new PadLayerParam();
        *param                     = layer_param;

        auto p = param_dict;
        int pad_t = INT_MIN;
        int pad_b = INT_MIN;
        int pad_l = INT_MIN;
        int pad_r = INT_MIN;

        pad_t = GetInt(p, 0, 0);
        pad_b = GetInt(p, 1, 0);
        pad_l = GetInt(p, 2, 0);
        pad_r = GetInt(p, 3, 0);

        layer_param->type = GetInt(p, p.size()-2, 0);
        layer_param->pads = {pad_t,pad_b,pad_l,pad_r};

        return TNN_OK;
    }

    Status PadLayerInterpreter::InterpretResource(Deserializer& deserializer, std::shared_ptr<LayerInfo> info,
                                                  LayerResource** resource) {
        return TNN_OK;                                    
    }

} // namespace ncnn

} // namespace TNN_NS