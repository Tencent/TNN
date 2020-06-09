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

namespace TNN_NS {

namespace ncnn {

    DECLARE_LAYER_INTERPRETER(BatchNorm);

    REGISTER_LAYER_INTERPRETER(BatchNorm, BatchNorm);

    Status BatchNormLayerInterpreter::InterpretProto(std::string type_name, str_dict param_dict, LayerType& type,
                                                     LayerParam** param) {
        BatchNormLayerParam* layer_param = new BatchNormLayerParam();
        *param                           = layer_param;

        type = ConvertNCNNLayerType(type_name);

        auto& p               = param_dict;
        layer_param->channels = GetInt(p, 0, 0);
        layer_param->eps      = GetFloat(p, 1, 0.0);

        return TNN_OK;
    }

    Status BatchNormLayerInterpreter::InterpretResource(Deserializer& deserializer, std::shared_ptr<LayerInfo> info,
                                                        LayerResource** resource) {
        BatchNormLayerResource* layer_res = new BatchNormLayerResource();
        *resource                         = layer_res;

        auto param = std::dynamic_pointer_cast<BatchNormLayerParam>(info->param);
        if (!param) {
            return Status(TNNERR_LAYER_ERR, "layer param is nil: BatchNormLayerParam");
        }

        RawBuffer slope;
        RawBuffer mean;
        RawBuffer var;
        RawBuffer bias;

        deserializer.GetRawSimple(slope, param->channels);
        deserializer.GetRawSimple(mean, param->channels);
        deserializer.GetRawSimple(var, param->channels);
        deserializer.GetRawSimple(bias, param->channels);

        int k_size_in_bytes = static_cast<int>(param->channels * sizeof(float));
        RawBuffer k(k_size_in_bytes);
        RawBuffer b(k_size_in_bytes);

        float* slope_data = slope.force_to<float*>();
        float* mean_data  = mean.force_to<float*>();
        float* var_data   = var.force_to<float*>();
        float* bias_data  = bias.force_to<float*>();
        float* k_data     = k.force_to<float*>();
        float* b_data     = b.force_to<float*>();
        float eps         = param->eps;

        for (int i = 0; i < param->channels; i++) {
            float sqrt_var = static_cast<float>(sqrt(var_data[i] + eps));
            k_data[i]      = slope_data[i] / sqrt_var;
            b_data[i]      = bias_data[i] - slope_data[i] * mean_data[i] / sqrt_var;
        }

        layer_res->scale_handle = k;
        layer_res->bias_handle  = b;

        return TNN_OK;
    }

}  // namespace ncnn

}  // namespace TNN_NS
