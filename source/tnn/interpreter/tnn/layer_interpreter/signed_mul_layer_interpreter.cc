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

#include "abstract_layer_interpreter.h"

namespace TNN_NS {

DECLARE_LAYER_INTERPRETER(SignedMul, LAYER_SIGNED_MUL);

    Status SignedMulLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int start_index, LayerParam** param) {
        int index = start_index;

        auto layer_param = new SignedMulLayerParam();
        *param                         = layer_param;

        layer_param->alpha = (float)atof(layer_cfg_arr[index++].c_str());
        layer_param->beta = (float)atof(layer_cfg_arr[index++].c_str());
        layer_param->gamma = (float)atof(layer_cfg_arr[index++].c_str());

        return TNN_OK;
    }

    Status SignedMulLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
        return TNN_OK;
    }

    Status SignedMulLayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
        auto layer_param = dynamic_cast<SignedMulLayerParam*>(param);

        if (nullptr == layer_param) {
            LOGE("invalid layer param to save\n");
            return Status(TNNERR_NULL_PARAM, "invalid layer param to save");
        }

        output_stream << layer_param->alpha << " ";
        output_stream << layer_param->beta << " ";
        output_stream << layer_param->gamma << " ";

        return TNN_OK;
    }

    Status SignedMulLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
        return TNN_OK;
    }

REGISTER_LAYER_INTERPRETER(SignedMul, LAYER_SIGNED_MUL);

} // namespace TNN_NS