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

    DECLARE_LAYER_INTERPRETER(Reshape);

    REGISTER_LAYER_INTERPRETER(Reshape, Reshape);

    Status ReshapeLayerInterpreter::InterpretProto(std::string type_name, str_dict param_dict, LayerType& type,
                                                   LayerParam** param) {
        ReshapeLayerParam* layer_param = new ReshapeLayerParam();
        *param                         = layer_param;

        type = ConvertNCNNLayerType(type_name);

        auto& p = param_dict;

        int w       = GetInt(p, 0, 0);
        int h       = GetInt(p, 1, 0);
        int c       = GetInt(p, 2, 0);
        int permute = GetInt(p, 3, 0);

        if (permute != 0) {
            return Status(TNNERR_INVALID_NETCFG, "ncnn reshape with permute is not supported now");
        }

        if (c == 0 && h == 0) {
            layer_param->shape = {0, w, 1, 1};
        } else if (c == 0) {
            layer_param->shape = {0, w, h, 1};
        } else {
            layer_param->shape = {0, c, h, w};
        }

        layer_param->axis     = 0;
        layer_param->num_axes = 4;

        return TNN_OK;
    }

    Status ReshapeLayerInterpreter::InterpretResource(Deserializer& deserializer, std::shared_ptr<LayerInfo> info,
                                                      LayerResource** resource) {
        return TNN_OK;
    }

}  // namespace ncnn

}  // namespace TNN_NS
