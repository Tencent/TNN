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

#include <tnn/utils/dims_vector_utils.h>

#include "tflite_op_converter.h"
#include "tflite_utils.h"

namespace TNN_CONVERTER {

DECLARE_OP_CONVERTER(Gather);

std::string TFLiteGatherConverter::TNNOpType(tflite::BuiltinOperator op_code, bool quantized_model) {
    return "Gather";
}

tflite::ActivationFunctionType TFLiteGatherConverter::ActivationType(
    const std::unique_ptr<tflite::OperatorT>& tf_lite_operator, tflite::BuiltinOperator op_code) {
    return tflite::ActivationFunctionType_NONE;
}

TNN_NS::Status TFLiteGatherConverter::exec(TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource,
                                           const std::unique_ptr<tflite::OperatorT>& tf_lite_operator,
                                           const std::vector<std::unique_ptr<tflite::TensorT>>& tf_lite_tensors,
                                           const std::vector<std::unique_ptr<tflite::BufferT>>& tf_lite_model_buffer,
                                           const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tf_lite_op_set,
                                           bool quantized_model) {
    auto param                    = new TNN_NS::GatherLayerParam;
    auto cur_layer                = net_structure.layers.back();
    cur_layer->param              = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->name                   = cur_layer->name;
    param->type                   = cur_layer->type_str;
    param->quantized              = quantized_model;
    auto option                   = tf_lite_operator->builtin_options.AsGatherOptions();
    const auto& input_tensor      = tf_lite_tensors[tf_lite_operator->inputs[0]];
    param->axis                   = ConvertAxisFormatTFLite(option->axis, input_tensor->shape.size());
    auto& resource_map            = net_resource.resource_map;
    auto resource                 = std::make_shared<TNN_NS::GatherLayerResource>();
    resource_map[cur_layer->name] = resource;
    const auto& input_data        = tf_lite_model_buffer[input_tensor->buffer]->data;
    if (!input_data.empty()) {
        param->data_in_resource = true;
        auto input_dims         = input_tensor->shape;
        if (input_dims.size() != 2) {
            LOGE("TNN TFLite Converter only support Gather's input dims size 2\n");
            return TNN_NS::TNNERR_CONVERT_UNSUPPORT_LAYER;
        }
        auto count                        = TNN_NS::DimsVectorUtils::Count(input_dims);
        TNN_NS::RawBuffer data_raw_buffer = TNN_NS::RawBuffer(count * sizeof(float), input_dims);
        data_raw_buffer.SetDataType(TNN_NS::DATA_TYPE_FLOAT);
        memcpy(data_raw_buffer.force_to<float*>(), input_data.data(), count * sizeof(float));
        resource->data = data_raw_buffer;
        cur_layer->inputs.erase(cur_layer->inputs.begin());
    } else {
        param->data_in_resource = false;
    }
    const auto& indices_tensor = tf_lite_tensors[tf_lite_operator->inputs[1]];
    const auto& indices_data = tf_lite_model_buffer[indices_tensor->buffer]->data;
    if (!indices_data.empty()) {
        auto indices_dims = input_tensor->shape;
        if (indices_dims.size() != 2) {
            LOGE("TNN TFLite Converter only support Gather's input dims size 2\n");
            return TNN_NS::TNNERR_CONVERT_UNSUPPORT_LAYER;
        }
        auto count = TNN_NS::DimsVectorUtils::Count(indices_dims);
        TNN_NS::RawBuffer indices_raw_buffer = TNN_NS::RawBuffer(count * sizeof(float ), indices_dims);
        indices_raw_buffer.SetDataType(TNN_NS::DATA_TYPE_FLOAT);
        memcpy(indices_raw_buffer.force_to<float*>(), indices_data.data(), count*sizeof(float));
        resource->indices = indices_raw_buffer;
        cur_layer->inputs.erase(cur_layer->inputs.begin() + 1);
    } else {
        param->indices_in_resource = false;
    }
    return TNN_NS::TNN_CONVERT_OK;
}
using namespace tflite;
REGISTER_CONVERTER(Gather, BuiltinOperator_GATHER);

}  // namespace TNN_CONVERTER