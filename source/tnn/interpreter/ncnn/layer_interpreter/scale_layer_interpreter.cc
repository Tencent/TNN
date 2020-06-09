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

    DECLARE_LAYER_INTERPRETER(Scale);

    REGISTER_LAYER_INTERPRETER(Scale, Scale);

    Status ScaleLayerInterpreter::InterpretProto(std::string type_name, str_dict param_dict, LayerType& type,
                                          LayerParam** param) {
        type = ConvertNCNNLayerType(type_name);
        ScaleLayerParam* layer_param = new ScaleLayerParam();
        *param                       = layer_param;

        auto p = param_dict;
        layer_param->axis = 1;
        layer_param->num_axes = 1;
        layer_param->bias_term = GetInt(p, 1, 0);
        layer_param->weight_data_size = GetInt(p, 0, 0);

        return TNN_OK;
    }

    Status ScaleLayerInterpreter::InterpretResource(Deserializer& deserializer, std::shared_ptr<LayerInfo> info,
                                                    LayerResource** resource) {
        BatchNormLayerResource* layer_res = new BatchNormLayerResource();
        *resource                         = layer_res;

        auto param = std::dynamic_pointer_cast<ScaleLayerParam>(info->param);
        if (!param) {
            return Status(TNNERR_LAYER_ERR, "Scale Layer Param is nil: ScaleLayerParam");
        }

        if (param->weight_data_size == -233) {
            return Status(TNNERR_PARAM_ERR, "Scale Layer Param is invalid: ScaleLayerParam");
        }

        layer_res->name = param->name;

        RawBuffer scale;
        deserializer.GetRawSimple(scale, param->weight_data_size);

        RawBuffer bias;
        if (param->bias_term) {
            deserializer.GetRawSimple(bias, param->weight_data_size);
        }

        layer_res->scale_handle = scale;
        layer_res->bias_handle = bias;

        return TNN_OK;
    }

}  // namespace ncnn

}  // namespace TNN_NS
