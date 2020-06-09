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

#include <cmath>

namespace TNN_NS {

namespace ncnn {

    DECLARE_LAYER_INTERPRETER(ReLU);

    REGISTER_LAYER_INTERPRETER(ReLU, ReLU);

    Status ReLULayerInterpreter::InterpretProto(std::string type_name, str_dict param_dict, LayerType& type,
                                                LayerParam** param) {
        type = ConvertNCNNLayerType(type_name);

        auto& p = param_dict;

        float slope = GetFloat(p, 0, 0.0);

        if (fabs(slope) > 1e-6) {
            type = LAYER_PRELU;

            PReluLayerParam* layer_param = new PReluLayerParam();
            *param                       = layer_param;

            layer_param->channel_shared = 1;
            float* ptr                  = reinterpret_cast<float*>(&layer_param->has_filler);
            ptr[0]                      = slope;
        }

        return TNN_OK;
    }

    Status ReLULayerInterpreter::InterpretResource(Deserializer& deserializer, std::shared_ptr<LayerInfo> info,
                                                   LayerResource** resource) {
        if (info->type == LAYER_PRELU) {
            auto param = std::dynamic_pointer_cast<PReluLayerParam>(info->param);
            if (!param) {
                return Status(TNNERR_LAYER_ERR, "layer param is nil: PReluLayerParam");
            }

            PReluLayerResource* layer_res = new PReluLayerResource();
            *resource                     = layer_res;

            RawBuffer slope(4);
            float* slope_data = slope.force_to<float*>();
            float* ptr        = reinterpret_cast<float*>(&param->has_filler);
            slope_data[0]     = ptr[0];

            layer_res->slope_handle = slope;
        }

        return TNN_OK;
    }

}  // namespace ncnn

}  // namespace TNN_NS
