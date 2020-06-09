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

    DECLARE_LAYER_INTERPRETER(InnerProduct);

    REGISTER_LAYER_INTERPRETER(InnerProduct, InnerProduct);

    Status InnerProductLayerInterpreter::InterpretProto(std::string type_name, str_dict param_dict, LayerType& type,
                                                        LayerParam** param) {
        type = ConvertNCNNLayerType(type_name);

        auto& p = param_dict;

        int num_output       = GetInt(p, 0, 0);
        int bias_term        = GetInt(p, 1, 0);
        int weight_data_size = GetInt(p, 2, 0);

        int int8_scale_term = GetInt(p, 8, 0);

        // Not supported yet
        int activation_type    = GetInt(p, 9, 0);
        auto activation_params = GetFloatList(p, 10);

        InnerProductLayerParam* layer_param = new InnerProductLayerParam();
        *param                              = layer_param;

        layer_param->num_output       = num_output;
        layer_param->has_bias         = bias_term;
        layer_param->transpose        = 0;  // TODO
        layer_param->axis             = 1;  // Default axies is w in ncnn;
        layer_param->weight_data_size = weight_data_size;

        return TNN_OK;
    }

    Status InnerProductLayerInterpreter::InterpretResource(Deserializer& deserializer, std::shared_ptr<LayerInfo> info,
                                                           LayerResource** resource) {
        InnerProductLayerResource* layer_res = new InnerProductLayerResource();
        *resource                            = layer_res;

        auto param = std::dynamic_pointer_cast<InnerProductLayerParam>(info->param);
        if (!param) {
            return Status(TNNERR_LAYER_ERR, "layer param is nil: InnerProductLayerParam");
        }

        RawBuffer weights;
        deserializer.GetRaw(weights, param->weight_data_size);
        layer_res->weight_handle = weights;

        if (param->has_bias) {
            RawBuffer bias;
            deserializer.GetRawSimple(bias, param->num_output);
            layer_res->bias_handle = bias;
        }

        // LOGDT("ip model %d %.6f\n", "ncnn", param->weight_data_size, weights.force_to<float *>()[0]);

        return TNN_OK;
    }

}  // namespace ncnn

}  // namespace TNN_NS
