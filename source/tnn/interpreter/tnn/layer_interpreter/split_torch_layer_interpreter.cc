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

#include "tnn/interpreter/tnn/layer_interpreter/abstract_layer_interpreter.h"

#include <stdlib.h>

namespace TNN_NS {

    DECLARE_LAYER_INTERPRETER(SplitTorch, LAYER_SPLITTORCH);

    Status SplitTorchLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int index, LayerParam** param) {
        auto p = CreateLayerParam<SplitTorchLayerParam>(param);

        int slice_count = 0;
        GET_INT_2(p->axis, slice_count);

        p->slices.clear();
        GET_INT_N_INTO_VEC(p->slices, slice_count);
        GET_INT_1_OR_DEFAULT(p->is_split_specified, 0);

        GET_INT_1(p->split_size);

        return TNN_OK;
    }

    Status SplitTorchLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
        return TNN_OK;
    }

    Status SplitTorchLayerInterpreter::SaveProto(std::ostream& output_stream, LayerParam* param) {
        CAST_OR_RET_ERROR(split_torch_param, SplitTorchLayerParam, "invalid layer param to save", param);

        output_stream << split_torch_param->axis << " ";
        output_stream << split_torch_param->slices.size() << " ";
        for (auto item : split_torch_param->slices) {
            output_stream << item << " ";
        }
        output_stream << int(split_torch_param->is_split_specified) << " ";
        output_stream << split_torch_param->split_size << " ";

        return TNN_OK;
    }

    Status SplitTorchLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
        return TNN_OK;
    }

    REGISTER_LAYER_INTERPRETER(SplitTorch, LAYER_SPLITTORCH);

}  // namespace TNN_NS
