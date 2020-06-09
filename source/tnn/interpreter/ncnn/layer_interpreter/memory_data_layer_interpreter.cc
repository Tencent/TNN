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

#include <vector>

namespace TNN_NS {

namespace ncnn {

    DECLARE_LAYER_INTERPRETER(MemoryData);

    REGISTER_LAYER_INTERPRETER(MemoryData, MemoryData);

    Status MemoryDataLayerInterpreter::InterpretProto(std::string type_name, str_dict param_dict, LayerType& type,
                                                      LayerParam** param) {
        ConstLayerParam* layer_param = new ConstLayerParam();
        *param                       = layer_param;

        type = ConvertNCNNLayerType(type_name);

        auto& p = param_dict;
        int w   = GetInt(p, 0, 0);
        int h   = GetInt(p, 1, 0);
        int c   = GetInt(p, 2, 0);

        std::vector<int> dims = {w, h, c};
        layer_param->dims.resize(0);
        for (auto i : dims) {
            if (i != 0) {
                layer_param->dims.push_back(i);
            }
        }

        if (layer_param->dims.size() == 0) {
            return Status(TNNERR_INVALID_NETCFG, "ncnn MemoryData param error.");
        }

        return TNN_OK;
    }

    Status MemoryDataLayerInterpreter::InterpretResource(Deserializer& deserializer, std::shared_ptr<LayerInfo> info,
                                                         LayerResource** resource) {
        ConstLayerResource* layer_res = new ConstLayerResource();
        *resource                     = layer_res;

        auto param = std::dynamic_pointer_cast<ConstLayerParam>(info->param);
        if (!param) {
            return Status(TNNERR_LAYER_ERR, "layer param is nil: ConstLayerParam");
        }

        int data_count = 1;
        for (int dim_i : param->dims) {
            data_count *= dim_i;
        }

        RawBuffer weight;
        deserializer.GetRawSimple(weight, data_count);

        layer_res->weight_handle = weight;

        return TNN_OK;
    }

}  // namespace ncnn

}  // namespace TNN_NS
