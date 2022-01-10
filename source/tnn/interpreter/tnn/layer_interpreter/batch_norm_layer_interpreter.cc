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

DECLARE_LAYER_INTERPRETER(BatchNorm, LAYER_BATCH_NORM);

Status BatchNormLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int index, LayerParam** param) {
        return TNN_OK;
}

Status BatchNormLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    auto batchnorm_res = CreateLayerRes<BatchNormLayerResource>(resource);

    GET_BUFFER_FOR_ATTR(batchnorm_res, scale_handle, deserializer);
    GET_BUFFER_FOR_ATTR(batchnorm_res, bias_handle, deserializer);

    if (batchnorm_res->bias_handle.GetBytesSize() == 0) {
        size_t scal_byte_size      = batchnorm_res->scale_handle.GetBytesSize();
        batchnorm_res->bias_handle = RawBuffer(scal_byte_size);
    }

    return TNN_OK;
}

Status BatchNormLayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
    return TNN_OK;
}

Status BatchNormLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    CAST_OR_RET_ERROR(batchnorm_res, BatchNormLayerResource, "invalid layer res to save", resource);
    serializer.PutRaw(batchnorm_res->scale_handle);
    serializer.PutRaw(batchnorm_res->bias_handle);
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(BatchNorm, LAYER_BATCH_NORM);

}  // namespace TNN_NS
