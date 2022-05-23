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

#include "math.h"

namespace TNN_NS {

DECLARE_LAYER_INTERPRETER(BatchNormEx, LAYER_BATCH_NORM_EX);

Status BatchNormExLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int start_index, LayerParam** param) {
    return TNN_OK;
}

Status BatchNormExLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    auto batchnorm_res = CreateLayerRes<BatchNormLayerResource>(resource);
    std::string layer_name = deserializer.GetString();
    int has_data3 = deserializer.GetInt();
    RawBuffer data1_buf, data2_buf, data3_buf;
    deserializer.GetRaw(data1_buf);
    deserializer.GetRaw(data2_buf);
    deserializer.GetRaw(data3_buf);
    float* data_1 = data1_buf.force_to<float*>();
    float* data_2 = data2_buf.force_to<float*>();
    float* data_3 = data3_buf.force_to<float*>();
    int channel = data1_buf.GetDataCount();
    for(int i = 0; i < channel; ++i) {
        // k = 1.0 / sqrt(var + 0.00001) 
        data_2[i] = 1.0f / pow(data_2[i]/data_3[0] + 0.00001f, 0.5f);
        // bias = -mean * k
        data_1[i] =  -data_1[i] * data_2[i] / data_3[0];
    }

    batchnorm_res->scale_handle = data2_buf;
    batchnorm_res->bias_handle = data1_buf;

    return TNN_OK;
}

Status BatchNormExLayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
    return TNN_OK;
}

Status BatchNormExLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(BatchNormEx, LAYER_BATCH_NORM_EX);

}  // namespace TNN_NS
