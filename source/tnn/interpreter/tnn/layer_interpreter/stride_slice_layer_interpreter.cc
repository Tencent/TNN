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
#include <algorithm>

namespace TNN_NS {

DECLARE_LAYER_INTERPRETER(StrideSlice, LAYER_STRIDED_SLICE);

Status StrideSliceLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int start_index, LayerParam** param) {
    StrideSliceLayerParam* layer_param = new StrideSliceLayerParam();
    *param                             = layer_param;
    int index                          = start_index;

    // Note: old order is n c h w,
    std::vector<int> begins;
    int n1 = atoi(layer_cfg_arr[index++].c_str());
    for (; n1 > 0; index++, n1--) {
        if (index < layer_cfg_arr.size()) {
            begins.push_back(atoi(layer_cfg_arr[index].c_str()));
        } else {
            LOGE("StrideSliceLayerInterpreter param is invalid\n");
            return Status(TNNERR_PARAM_ERR, "StrideSliceLayerInterpreter param is invalid");
        }
    }
    std::reverse(begins.begin(), begins.end());
    layer_param->begins = begins;

    std::vector<int> ends;
    n1 = atoi(layer_cfg_arr[index++].c_str());
    for (; n1 > 0; index++, n1--) {
        if (index < layer_cfg_arr.size()) {
            ends.push_back(atoi(layer_cfg_arr[index].c_str()));
        } else {
            LOGE("StrideSliceLayerInterpreter param is invalid\n");
            return Status(TNNERR_PARAM_ERR, "StrideSliceLayerInterpreter param is invalid");
        }
    }
    std::reverse(ends.begin(), ends.end());
    layer_param->ends = ends;

    std::vector<int> strides;
    n1 = atoi(layer_cfg_arr[index++].c_str());
    for (; n1 > 0; index++, n1--) {
        if (index < layer_cfg_arr.size()) {
            strides.push_back(atoi(layer_cfg_arr[index].c_str()));
        } else {
            LOGE("StrideSliceLayerInterpreter param is invalid\n");
            return Status(TNNERR_PARAM_ERR, "StrideSliceLayerInterpreter param is invalid");
        }
    }
    std::reverse(strides.begin(), strides.end());
    layer_param->strides = strides;
    return TNN_OK;
}

Status StrideSliceLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    return TNN_OK;
}

Status StrideSliceLayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
    auto layer_param = dynamic_cast<StrideSliceLayerParam*>(param);
    if (nullptr == layer_param) {
        LOGE("invalid layer param to save\n");
        return Status(TNNERR_NULL_PARAM, "invalid layer param to save");
    }
    auto begins = layer_param->begins;
    std::reverse(begins.begin(), begins.end());
    output_stream << begins.size() << " ";
    for (int i = 0; i < begins.size(); i++) {
        output_stream << begins[i] << " ";
    }

    auto ends = layer_param->ends;
    std::reverse(ends.begin(), ends.end());
    output_stream << ends.size() << " ";
    for (int i = 0; i < ends.size(); i++) {
        output_stream << ends[i] << " ";
    }

    auto strides = layer_param->strides;
    std::reverse(strides.begin(), strides.end());
    output_stream << strides.size() << " ";
    for (int i = 0; i < strides.size(); i++) {
        output_stream << strides[i] << " ";
    }
    return TNN_OK;
}

Status StrideSliceLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(StrideSlice, LAYER_STRIDED_SLICE);

}  // namespace TNN_NS
