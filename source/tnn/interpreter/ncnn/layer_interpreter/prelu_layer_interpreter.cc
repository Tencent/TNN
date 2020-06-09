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

    DECLARE_LAYER_INTERPRETER(PRelu);

    REGISTER_LAYER_INTERPRETER(PRelu, PReLU);

    Status PReluLayerInterpreter::InterpretProto(std::string type_name, str_dict param_dict, LayerType& type,
                                                 LayerParam** param) {
        type = ConvertNCNNLayerType(type_name);

        PReluLayerParam* layer_param = new PReluLayerParam();
        *param                       = layer_param;

        auto& p = param_dict;
        layer_param->channel_shared = GetInt(p, 0, 0) == 1 ? 1 : 0;
        layer_param->has_filler = 0;

        layer_param->weight_data_size = GetInt(p, 0, 1); // PReLU resource byteSize

        return TNN_OK;
    }

    Status PReluLayerInterpreter::InterpretResource(Deserializer& deserializer, std::shared_ptr<LayerInfo> info,
                                                   LayerResource** resource) {
        PReluLayerResource* layer_res = new PReluLayerResource();
        *resource                     = layer_res;

        auto param = std::dynamic_pointer_cast<PReluLayerParam>(info->param);
        if (!param) {
            return Status(TNNERR_LAYER_ERR, "PReLU layer param is nil: PreluLayerParam");
        }
        
        layer_res->name = param->name;
    
        RawBuffer k;
        deserializer.GetRawSimple(k, param->weight_data_size);
        layer_res->slope_handle = k;

        return TNN_OK;
    }

}  // namspace ncnn

}  // namespace TNN_NS