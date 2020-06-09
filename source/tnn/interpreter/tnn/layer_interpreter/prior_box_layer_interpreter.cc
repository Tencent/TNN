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

DECLARE_LAYER_INTERPRETER(PriorBox, LAYER_PRIOR_BOX);

Status PriorBoxLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int start_index, LayerParam **param) {
    auto layer_param = new PriorBoxLayerParam();
    *param           = layer_param;
    int index        = start_index;

    // get min_size
    int min_size_count = atoi(layer_cfg_arr[index++].c_str());
    for (int i = 0; i < min_size_count; ++i) {
        layer_param->min_sizes.push_back(static_cast<float>(atof(layer_cfg_arr[index++].c_str())));
    }
    // get max_size
    int max_size_count = atoi(layer_cfg_arr[index++].c_str());
    for (int i = 0; i < max_size_count; ++i) {
        layer_param->max_sizes.push_back(static_cast<float>(atof(layer_cfg_arr[index++].c_str())));
    }

    // get clip
    if (atoi(layer_cfg_arr[index++].c_str()) == 1) {
        layer_param->clip = true;
    } else {
        layer_param->clip = false;
    }

    // get flip
    if (atoi(layer_cfg_arr[index++].c_str()) == 1) {
        layer_param->flip = true;
    } else {
        layer_param->flip = false;
    }

    // get variance
    int variance_count = atoi(layer_cfg_arr[index++].c_str());
    for (int i = 0; i < variance_count; ++i) {
        layer_param->variances.push_back(static_cast<float>(atof(layer_cfg_arr[index++].c_str())));
    }

    // get aspect_ratio
    int aspect_ratios_count = atoi(layer_cfg_arr[index++].c_str());
    for (int i = 0; i < aspect_ratios_count; ++i) {
        layer_param->aspect_ratios.push_back(static_cast<float>(atof(layer_cfg_arr[index++].c_str())));
    }
    // get img_sizes
    layer_param->img_w = atoi(layer_cfg_arr[index++].c_str());
    layer_param->img_h = atoi(layer_cfg_arr[index++].c_str());
    // get step
    layer_param->step_w = atoi(layer_cfg_arr[index++].c_str());
    layer_param->step_h = atoi(layer_cfg_arr[index++].c_str());
    // get offset
    layer_param->offset = static_cast<float>(atof(layer_cfg_arr[index++].c_str()));

    return TNN_OK;
}

Status PriorBoxLayerInterpreter::InterpretResource(Deserializer &deserializer, LayerResource **Resource) {
    return TNN_OK;
}

Status PriorBoxLayerInterpreter::SaveProto(std::ofstream &output_stream, LayerParam *param) {
    PriorBoxLayerParam *layer_param = dynamic_cast<PriorBoxLayerParam *>(param);
    if (nullptr == layer_param) {
        LOGE("invalid layer param to save\n");
        return Status(TNNERR_NULL_PARAM, "invalid layer param to save");
    }

    // write min_size
    output_stream << layer_param->min_sizes.size() << " ";
    for (float min_size : layer_param->min_sizes) {
        output_stream << min_size << " ";
    }

    // write max_size
    output_stream << layer_param->max_sizes.size() << " ";
    for (float max_size : layer_param->max_sizes) {
        output_stream << max_size << " ";
    }

    // write clip
    output_stream << (layer_param->clip ? 1 : 0) << " ";

    // write flip
    output_stream << (layer_param->flip ? 1 : 0) << " ";

    // write variances
    output_stream << layer_param->variances.size() << " ";
    for (float variance : layer_param->variances) {
        output_stream << variance << " ";
    }

    // write aspect_ratio
    output_stream << layer_param->aspect_ratios.size() << " ";
    for (float aspect_ratio : layer_param->aspect_ratios) {
        output_stream << aspect_ratio << " ";
    }

    // write img_size : order img_size[img_w, img_h]
    output_stream << layer_param->img_w << " ";
    output_stream << layer_param->img_h << " ";

    // write step
    output_stream << layer_param->step_w << " ";
    output_stream << layer_param->step_h << " ";

    // write offset
    output_stream << layer_param->offset << " ";
    return TNN_OK;
}

Status PriorBoxLayerInterpreter::SaveResource(Serializer &serializer, LayerParam *param, LayerResource *resource) {
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(PriorBox, LAYER_PRIOR_BOX);

}  // namespace TNN_NS
