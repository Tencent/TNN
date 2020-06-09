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

DECLARE_LAYER_INTERPRETER(RoiPooling, LAYER_ROIPOOLING);

Status RoiPoolingLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int start_index, LayerParam** param) {
    RoiPoolingLayerParam* layer_param = new RoiPoolingLayerParam();
    *param                            = layer_param;
    int index                         = start_index;

    // pool_type
    layer_param->pool_type = atoi(layer_cfg_arr[index++].c_str());

    // spatial_scale
    layer_param->spatial_scale = static_cast<float>(atof(layer_cfg_arr[index++].c_str()));

    // pooled_dims
    int pooled_w = atoi(layer_cfg_arr[index++].c_str());
    int pooled_h = atoi(layer_cfg_arr[index++].c_str());

    layer_param->pooled_dims.push_back(pooled_w);
    layer_param->pooled_dims.push_back(pooled_h);

    if (index < layer_cfg_arr.size()) {
        int pooled_d = atoi(layer_cfg_arr[index++].c_str());
        layer_param->pooled_dims.push_back(pooled_d);
    }

    return TNN_OK;
}

Status RoiPoolingLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    return TNN_OK;
}

Status RoiPoolingLayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
    RoiPoolingLayerParam* layer_param = dynamic_cast<RoiPoolingLayerParam*>(param);
    if (nullptr == layer_param) {
        LOGE("invalid layer param to save\n");
        return Status(TNNERR_NULL_PARAM, "invalid layer param to save");
    }

    output_stream << layer_param->pool_type << " ";
    output_stream << layer_param->spatial_scale << " ";

    ASSERT(layer_param->pooled_dims.size() == 2);
    output_stream << layer_param->pooled_dims[0] << " ";
    output_stream << layer_param->pooled_dims[1] << " ";

    for (auto item : layer_param->pooled_dims)
        output_stream << item << " ";

    return TNN_OK;
}

Status RoiPoolingLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(RoiPooling, LAYER_ROIPOOLING);

}  // namespace TNN_NS
