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

#include "reduce_op_interpreter.h"
namespace TNN_NS {

Status ReduceOpLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int start_index, LayerParam **param) {
    auto *layer_param = new ReduceLayerParam();
    *param            = layer_param;
    int index         = start_index;

    layer_param->keep_dims = atoi(layer_cfg_arr[index++].c_str());

    layer_param->axis.clear();
    for (int i = index; i < layer_cfg_arr.size(); ++i) {
        int axis = atoi(layer_cfg_arr[index++].c_str());
        layer_param->axis.push_back(axis);
    }
    return TNN_OK;
}

Status ReduceOpLayerInterpreter::SaveProto(std::ofstream &output_stream, LayerParam *param) {
    auto *layer_param = dynamic_cast<ReduceLayerParam *>(param);
    if (nullptr == layer_param) {
        LOGE("invalid layer param to save\n");
        return Status(TNNERR_NULL_PARAM, "invalid layer param to save");
    }

    output_stream << layer_param->keep_dims << " ";
    //ASSERT(layer_param->axis.size() == 1);
    for(auto axis : layer_param->axis) {
        output_stream << axis << " ";
    }
    return TNN_OK;
}
}  // namespace TNN_NS

REGISTER_REDUCE_OP_LAYER_INTERPRETER(ReduceL1, LAYER_REDUCE_L1)

REGISTER_REDUCE_OP_LAYER_INTERPRETER(ReduceL2, LAYER_REDUCE_L2)

REGISTER_REDUCE_OP_LAYER_INTERPRETER(ReduceLogSum, LAYER_REDUCE_LOG_SUM)

REGISTER_REDUCE_OP_LAYER_INTERPRETER(ReduceLogSumExp, LAYER_REDUCE_LOG_SUM_EXP)

REGISTER_REDUCE_OP_LAYER_INTERPRETER(ReduceMax, LAYER_REDUCE_MAX)

REGISTER_REDUCE_OP_LAYER_INTERPRETER(ReduceMean, LAYER_REDUCE_MEAN)

REGISTER_REDUCE_OP_LAYER_INTERPRETER(ReduceMin, LAYER_REDUCE_MIN)

REGISTER_REDUCE_OP_LAYER_INTERPRETER(ReduceProd, LAYER_REDUCE_PROD)

REGISTER_REDUCE_OP_LAYER_INTERPRETER(ReduceSum, LAYER_REDUCE_SUM)

REGISTER_REDUCE_OP_LAYER_INTERPRETER(ReduceSumSquare, LAYER_REDUCE_SUM_SQUARE)