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

DECLARE_LAYER_INTERPRETER(BlobScale, LAYER_BLOB_SCALE);

Status BlobScaleLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int start_index, LayerParam** param) {
    return TNN_OK;
}

Status BlobScaleLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    auto layer_res = CreateLayerRes<IntScaleResource>(resource);

    // Use the DataType of first RawBuffer to distinguish the old and new versions
    // old version: scale_handle(float)
    // new version: zero_point_handle(int8), scale_handle(float)
    RawBuffer first_buffer;
    deserializer.GetRaw(first_buffer);
    if (first_buffer.GetDataType() == DATA_TYPE_INT8) {
        layer_res->zero_point_handle = first_buffer;
        GET_BUFFER_FOR_ATTR(layer_res, scale_handle, deserializer);
    } else if (first_buffer.GetDataType() == DATA_TYPE_FLOAT) {
        layer_res->scale_handle = first_buffer;
        int total_byte_size     = first_buffer.GetDataCount() * sizeof(char);
        RawBuffer zero_point_buffer(total_byte_size);
        zero_point_buffer.SetDataType(DATA_TYPE_INT8);
        memset(zero_point_buffer.force_to<int8_t*>(), 0, total_byte_size);
        layer_res->zero_point_handle = zero_point_buffer;
    } else {
        LOGE("invalid quantized layer Resource\n");
        return -1;
    }

    GET_BUFFER_FOR_ATTR(layer_res, bias_handle, deserializer);

    return TNN_OK;
}

Status BlobScaleLayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
    return TNN_OK;
}

Status BlobScaleLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    CAST_OR_RET_ERROR(layer_res, IntScaleResource, "invalid blob_scale to save", resource);

    // put zero_point_handle in front of scale_handle to distinguish the old and new versions
    serializer.PutRaw(layer_res->zero_point_handle);
    serializer.PutRaw(layer_res->scale_handle);
    serializer.PutRaw(layer_res->bias_handle);

    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(BlobScale, LAYER_BLOB_SCALE);

}  // namespace TNN_NS
