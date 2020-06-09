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

    DECLARE_LAYER_INTERPRETER(Interp);

    REGISTER_LAYER_INTERPRETER(Interp, Interp);

    Status InterpLayerInterpreter::InterpretProto(std::string type_name, str_dict param_dict, LayerType& type,
                                                  LayerParam** param) {
        type = ConvertNCNNLayerType(type_name);

        UpsampleLayerParam* layer_param = new UpsampleLayerParam();
        *param                          = layer_param;

        auto& p = param_dict;

        /* ncnn resize_type:
            1  : nearest
            2  : bilinear
            3  : bicubic
        */
        int resize_type    = GetInt(p, 0, 0);
        float height_scale = GetFloat(p, 1, 1.f);
        float width_scale  = GetFloat(p, 2, 1.f);
        int output_height  = GetInt(p, 3, 0);
        int output_width   = GetInt(p, 4, 0);

        // only supports nearest and bilinear now
        if (resize_type != 1 && resize_type != 2) {
            return Status(TNNERR_INVALID_NETCFG, "Interp layer: unsupported resize_type");
        }

        layer_param->type          = resize_type;
        layer_param->align_corners = 0;
        layer_param->scales.push_back(width_scale);
        layer_param->scales.push_back(height_scale);
        if (output_height != 0 && output_width != 0) {
            layer_param->dims.push_back(output_width);
            layer_param->dims.push_back(output_height);
        }

        return TNN_OK;
    }

    Status InterpLayerInterpreter::InterpretResource(Deserializer& deserializer, std::shared_ptr<LayerInfo> info,
                                                     LayerResource** resource) {
        return TNN_OK;
    }

}  // namespace ncnn

}  // namespace TNN_NS
