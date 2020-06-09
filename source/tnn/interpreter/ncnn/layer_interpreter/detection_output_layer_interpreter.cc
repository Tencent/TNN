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

    DECLARE_LAYER_INTERPRETER(DetectionOutput);

    REGISTER_LAYER_INTERPRETER(DetectionOutput, DetectionOutput);

    Status DetectionOutputLayerInterpreter::InterpretProto(std::string type_name, str_dict param_dict, LayerType& type,
                                                           LayerParam** param) {
        type = ConvertNCNNLayerType(type_name);

        DetectionOutputLayerParam* layer_param = new DetectionOutputLayerParam();
        *param                                 = layer_param;

        auto p = param_dict;

        int num_class = GetInt(p, 0, 0);
        layer_param->num_classes = num_class;

        layer_param->share_location = true; // ncnn does not have this controlled param
        
        layer_param->variance_encoded_in_target = num_class == -233 ? true : false; // 
        layer_param->code_type = 2; // code_type == PriorBoxParameter_CodeType_CENTER_SIZE
        layer_param->nms_param.nms_threshold = GetFloat(p, 1, 0.05f);
        layer_param->nms_param.top_k = GetInt(p, 2, 300);
        layer_param->keep_top_k = GetInt(p, 3, 100);
        layer_param->confidence_threshold = GetFloat(p, 4, 0.5f);
        layer_param->eta = 1.0f; // eta < 1.0 will enter a tnn branch while ncnn does not
        layer_param->background_label_id = 0; // tnn will skip background_label_id class from 0, ncnn does that from 1
        float variance = GetFloat(p, 5, -0.2f);
        if (variance != -0.2f && num_class == -233) {
            return Status(TNNERR_LAYER_ERR, "DetectionOutput Param is invalid: DetectionOutputLayerParam");
            // this means ncnn will use variances param in param list while tnn does not have that
        }

        return TNN_OK;
    }

    Status DetectionOutputLayerInterpreter::InterpretResource(Deserializer& deserializer, std::shared_ptr<LayerInfo> info,
                                                              LayerResource** resource) {
        return TNN_OK;
    }

}  // namespace ncnn

}  // namespace TNN_NS