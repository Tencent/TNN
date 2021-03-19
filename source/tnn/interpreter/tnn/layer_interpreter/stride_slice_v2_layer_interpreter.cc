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
#include <stdlib.h>

#include <algorithm>

#include "tnn/interpreter/tnn/layer_interpreter/abstract_layer_interpreter.h"

namespace TNN_NS {

DECLARE_LAYER_INTERPRETER(StrideSliceV2, LAYER_STRIDED_SLICE_V2);

Status StrideSliceV2LayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int start_index, LayerParam** param) {
    auto layer_param = new StrideSliceV2LayerParam();
    *param           = layer_param;
    int index        = start_index;

    std::vector<int> begins;
    int begin_sizes = atoi(layer_cfg_arr[index++].c_str());
    for (int i = 0; i < begin_sizes; ++i) {
        if (index < layer_cfg_arr.size()) {
            begins.push_back(atoi(layer_cfg_arr[index++].c_str()));
        } else {
            LOGE("StrideSliceV2LayerInterpreter param is invalid\n");
            return Status(TNNERR_PARAM_ERR, "StrideSliceV2LayerInterpreter param is invalid");
        }
    }
    layer_param->begins = begins;

    std::vector<int> ends;
    int end_size = atoi(layer_cfg_arr[index++].c_str());
    for (int i = 0; i < end_size; ++i) {
        if (index < layer_cfg_arr.size()) {
            ends.push_back(atoi(layer_cfg_arr[index++].c_str()));
        } else {
            LOGE("StrideSliceV2LayerInterpreter param is invalid\n");
            return Status(TNNERR_PARAM_ERR, "StrideSliceV2LayerInterpreter param is invalid");
        }
    }
    layer_param->ends = ends;

    std::vector<int> axes;
    int axes_size = atoi(layer_cfg_arr[index++].c_str());
    for (int i = 0; i < axes_size; ++i) {
        if (index < layer_cfg_arr.size()) {
            axes.push_back(atoi(layer_cfg_arr[index++].c_str()));
        } else {
            LOGE("StrideSliceV2LayerInterpreter param is invalid\n");
            return Status(TNNERR_PARAM_ERR, "StrideSliceV2LayerInterpreter param is invalid");
        }
    }
    layer_param->axes = axes;

    std::vector<int> strides;
    int stride_size = atoi(layer_cfg_arr[index++].c_str());
    for (int i = 0; i < stride_size; i++) {
        if (index < layer_cfg_arr.size()) {
            strides.push_back(atoi(layer_cfg_arr[index++].c_str()));
        } else {
            LOGE("StrideSliceV2LayerInterpreter param is invalid\n");
            return Status(TNNERR_PARAM_ERR, "StrideSliceV2LayerInterpreter param is invalid");
        }
    }
    layer_param->strides = strides;
    return TNN_OK;
}

Status StrideSliceV2LayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    return TNN_OK;
}

Status StrideSliceV2LayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
    auto layer_param = dynamic_cast<StrideSliceV2LayerParam*>(param);
    if (nullptr == layer_param) {
        LOGE("invalid layer param to save\n");
        return TNNERR_NULL_PARAM;
    }
    const auto& begins  = layer_param->begins;
    const auto& ends    = layer_param->ends;
    const auto& axes    = layer_param->axes;
    const auto& strides = layer_param->strides;
    output_stream << begins.size() << " ";
    for (const auto& begin : begins) {
        output_stream << begin << " ";
    }
    output_stream << ends.size() << " ";
    for (const auto& end : ends) {
        output_stream << end << " ";
    }
    output_stream << axes.size() << " ";
    for (const auto& axis : axes) {
        output_stream << axis << " ";
    }
    output_stream << strides.size() << " ";
    for (const auto& stride : strides) {
        output_stream << stride << " ";
    }
    return TNN_OK;
}

Status StrideSliceV2LayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(StrideSliceV2, LAYER_STRIDED_SLICE_V2);

}  // namespace TNN_NS
