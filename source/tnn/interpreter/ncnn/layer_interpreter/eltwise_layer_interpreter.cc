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

    DECLARE_LAYER_INTERPRETER(Eltwise);

    REGISTER_LAYER_INTERPRETER(Eltwise, Eltwise);

    static std::map<int, LayerType> global_elementwise_layer_type_map = {
        // NCNN Operation_PROD = 0, Operation_SUM = 1, Operation_MAX = 2
        {0, LAYER_MUL},
        {1, LAYER_ADD},
        {2, LAYER_MAXIMUM},
    };

    Status EltwiseLayerInterpreter::InterpretProto(std::string type_name, str_dict param_dict, LayerType& type,
                                                   LayerParam** param) {
        MultidirBroadcastLayerParam* layer_param = new MultidirBroadcastLayerParam();
        *param                                   = layer_param;

        auto& p = param_dict;

        int op_type             = GetInt(p, 0, 0);
        std::vector<float> coef = GetFloatList(p, 1);

        if (op_type < 0 || op_type > 2) {
            return Status(TNNERR_INVALID_NETCFG, "ncnn eltwise got invalid op_type");
        }

        type = global_elementwise_layer_type_map[op_type];

        if (coef.size() != 0) {
            return Status(TNNERR_INVALID_NETCFG, "ncnn eltwise layer with coefs is not supported now.");
        }

        return TNN_OK;
    }

    Status EltwiseLayerInterpreter::InterpretResource(Deserializer& deserializer, std::shared_ptr<LayerInfo> info,
                                                      LayerResource** resource) {
        return TNN_OK;
    }

}  // namespace ncnn

}  // namespace TNN_NS
