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

namespace TNN_NS {

DECLARE_LAYER_INTERPRETER(HdrGuide, LAYER_HDRGUIDE);

Status HdrGuideLayerInterpreter::InterpretProto(str_arr, int, LayerParam**) {
    return TNN_OK;
}

Status HdrGuideLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    HdrGuideLayerResource* layer_res = new HdrGuideLayerResource();
    *resource                        = layer_res;

    RawBuffer ccm_weight;
    deserializer.GetRaw(ccm_weight);
    layer_res->ccm_weight_handle = ccm_weight;

    RawBuffer ccm_bias;
    deserializer.GetRaw(ccm_bias);
    layer_res->ccm_bias_handle = ccm_bias;

    RawBuffer shifts;
    deserializer.GetRaw(shifts);
    layer_res->shifts_handle = shifts;

    RawBuffer slopes;
    deserializer.GetRaw(slopes);
    layer_res->slopes_handle = slopes;

    RawBuffer projection_weight;
    deserializer.GetRaw(projection_weight);
    layer_res->projection_weight_handle = projection_weight;

    RawBuffer projection_bias;
    deserializer.GetRaw(projection_bias);
    layer_res->projection_bias_handle = projection_bias;

    return TNN_OK;
}

Status HdrGuideLayerInterpreter::SaveProto(std::ofstream&, LayerParam*) {
    return TNN_OK;
}

Status HdrGuideLayerInterpreter::SaveResource(Serializer& serializer, LayerParam*, LayerResource* resource) {
    HdrGuideLayerResource* layer_res = dynamic_cast<HdrGuideLayerResource*>(resource);
    if (nullptr == layer_res) {
        LOGE("invalid layer res to save\n");
        return Status(TNNERR_NULL_PARAM, "invalid layer res to save");
    }

    serializer.PutRaw(layer_res->ccm_weight_handle);
    serializer.PutRaw(layer_res->ccm_bias_handle);
    serializer.PutRaw(layer_res->shifts_handle);
    serializer.PutRaw(layer_res->slopes_handle);
    serializer.PutRaw(layer_res->projection_weight_handle);
    serializer.PutRaw(layer_res->projection_bias_handle);

    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(HdrGuide, LAYER_HDRGUIDE);

}  // namespace TNN_NS
