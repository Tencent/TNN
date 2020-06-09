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

    DECLARE_LAYER_INTERPRETER(PriorBox);

    REGISTER_LAYER_INTERPRETER(PriorBox, PriorBox);

    Status PriorBoxLayerInterpreter::InterpretProto(std::string type_name, str_dict param_dict, LayerType& type,
                                                     LayerParam** param) {
        PriorBoxLayerParam* layer_param = new PriorBoxLayerParam();
        *param                           = layer_param;

        type = ConvertNCNNLayerType(type_name);

        auto& p               = param_dict;
        layer_param->min_sizes = GetFloatList(p, 0);
        layer_param->max_sizes =  GetFloatList(p, 1);
        layer_param->flip =  GetInt(p, 7, 1);
        layer_param->clip = GetInt(p, 8, 0);

        float variances[4];
        variances[0]=GetFloat(p, 3, 0.1f);
        variances[1]=GetFloat(p, 4, 0.1f);
        variances[2]=GetFloat(p, 5, 0.2f);
        variances[3]=GetFloat(p, 6, 0.2f);
        
        layer_param->variances.push_back(variances[0]);
        layer_param->variances.push_back(variances[1]);
        layer_param->variances.push_back(variances[2]);
        layer_param-> variances.push_back(variances[3]);


        layer_param->aspect_ratios =  GetFloatList(p, 2);
        layer_param->img_w =GetInt(p, 9, 0);
        layer_param->img_h = GetInt(p, 10, 0);
        layer_param->step_w = GetFloat(p, 11, -233.f);
        layer_param->step_h = GetFloat(p, 12, -233.f);
        layer_param->offset = GetFloat(p, 13, 0.f);
        //layer_param->step_mmdetection =GetInt(p, 14, 0);
        //layer_param->center_mmdetection =GetInt(p, 15, 0);

        return TNN_OK;
    }

    Status PriorBoxLayerInterpreter::InterpretResource(Deserializer& deserializer, std::shared_ptr<LayerInfo> info,
                                                        LayerResource** resource) {

        return TNN_OK;
    }

}  // namespace ncnn

}  // namespace TNN_NS
