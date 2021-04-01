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

DECLARE_LAYER_INTERPRETER(Upsample, LAYER_UPSAMPLE);

    Status UpsampleLayerInterpreter::InterpretProto(str_arr layer_cfg_arr,
                                                    int start_index,
                                                    LayerParam** param) {
        UpsampleLayerParam* layer_param = new UpsampleLayerParam();
        *param                         = layer_param;
        int index                      = start_index;

        // pool_type
        layer_param->mode = atoi(layer_cfg_arr[index++].c_str());

        // scales
        float scale_h = (float)atof(layer_cfg_arr[index++].c_str());
        float scale_w = (float)atof(layer_cfg_arr[index++].c_str());
        layer_param->scales.push_back(scale_w);
        layer_param->scales.push_back(scale_h);

        layer_param->align_corners = 0;
        if (index < layer_cfg_arr.size()) {
            layer_param->align_corners = atoi(layer_cfg_arr[index++].c_str());
        }

        // size
        int width, height;
        if ((index + 1) < layer_cfg_arr.size()) {
            height = atoi(layer_cfg_arr[index++].c_str());
            width  = atoi(layer_cfg_arr[index++].c_str());
            layer_param->dims.push_back(width);
            layer_param->dims.push_back(height);
        }
        return TNN_OK;
    }

    Status UpsampleLayerInterpreter::InterpretResource(
        Deserializer& deserializer, LayerResource** resource) {
        return TNN_OK;
    }

    Status UpsampleLayerInterpreter::SaveProto(std::ofstream& output_stream,
                                               LayerParam* param) {
        UpsampleLayerParam* layer_param =
            dynamic_cast<UpsampleLayerParam*>(param);
        if (nullptr == layer_param) {
            LOGE("invalid layer param to save\n");
            return Status(TNNERR_NULL_PARAM, "invalid layer param to save");
        }

        output_stream << layer_param->mode << " ";

        /*ASSERT(layer_param->scales.size() == 2);
        output_stream << layer_param->scales[1] << " ";
        output_stream << layer_param->scales[0] << " ";*/
        for(int i = layer_param->scales.size() - 1; i >= 0; i--) {
            output_stream << layer_param->scales[i] << " ";
        }

        output_stream << layer_param->align_corners << " ";

        if (layer_param->dims.size() == 2) {
            output_stream << layer_param->dims[1] << " ";
            output_stream << layer_param->dims[0] << " ";
        }

        return TNN_OK;
    }

    Status UpsampleLayerInterpreter::SaveResource(Serializer& serializer,
                                                  LayerParam* param,
                                                  LayerResource* resource) {
        return TNN_OK;
    }

REGISTER_LAYER_INTERPRETER(Upsample, LAYER_UPSAMPLE);

}  // namespace TNN_NS


