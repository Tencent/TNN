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

namespace ncnn    {

    DECLARE_LAYER_INTERPRETER(ReduceOp);

    REGISTER_LAYER_INTERPRETER(ReduceOp, Reduction);

    Status ReduceOpLayerInterpreter::InterpretProto(std::string type_name, str_dict param_dict, LayerType& type,
                                                     LayerParam** param) {
        ReduceLayerParam *layer_param = new ReduceLayerParam();
        *param                        = layer_param;
        
        static std::map<int, LayerType> reductipon_layer_type_map = {
            {0, LAYER_REDUCE_SUM},      {1, LAYER_NOT_SUPPORT},     {2, LAYER_REDUCE_SUM_SQUARE},
            {3, LAYER_REDUCE_MEAN},     {4, LAYER_REDUCE_MAX},      {5, LAYER_REDUCE_MIN},
            {6, LAYER_REDUCE_PROD},     {7, LAYER_NOT_SUPPORT},       {8, LAYER_REDUCE_L2},
            {9, LAYER_REDUCE_LOG_SUM},  {10,LAYER_REDUCE_LOG_SUM_EXP}
        }; // ncnn reduction map

        auto p = param_dict;
        int op_type = GetInt(p, 0, 0);
        type = reductipon_layer_type_map[op_type];
        
        int keep_dims = GetInt(p, 4, 0);
        std::vector<int> axis = GetIntList(p, 3);
        int reduce_all = GetInt(p, 1, 1);

        layer_param->keep_dims = keep_dims;
        layer_param->axis.assign(axis.begin(),axis.end());
        layer_param->all_reduce = reduce_all;
        
        return TNN_OK;
    }

    Status ReduceOpLayerInterpreter::InterpretResource(Deserializer& deserializer, std::shared_ptr<LayerInfo> info,
                                                       LayerResource** resource) {
        return TNN_OK;
    }

} // namespace ncnn

} // namespace TNN_NS
