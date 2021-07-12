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

#include <tnn/utils/data_format_converter.h>
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
        if (input_dims.size() <= 2) {
            auto count                        = TNN_NS::DimsVectorUtils::Count(input_dims);
            TNN_NS::RawBuffer data_raw_buffer = TNN_NS::RawBuffer(count * sizeof(float), input_dims);
            data_raw_buffer.SetDataType(TNN_NS::DATA_TYPE_FLOAT);
            memcpy(data_raw_buffer.force_to<float*>(), input_data.data(), count * sizeof(float));
            resource->data = data_raw_buffer;
            cur_layer->inputs.erase(cur_layer->inputs.begin());
        } else if (input_dims.size() == 3 || input_dims.size() == 4) {
            int n, c, h, w;
            if (input_dims.size() == 3) {
                n = input_dims[0];
                h = input_dims[1];
                w = 1;
                c = input_dims[2];
            } else {
                n = input_dims[0];
                h = input_dims[1];
                w = input_dims[2];
                c = input_dims[3];
            }
            const int count                   = TNN_NS::DimsVectorUtils::Count(input_dims);
            TNN_NS::RawBuffer data_raw_buffer = TNN_NS::RawBuffer(count * sizeof(float), input_dims);
            data_raw_buffer.SetDataType(TNN_NS::DATA_TYPE_FLOAT);
            auto tmp_buffer = new float[count]();
            TNN_NS::DataFormatConverter::ConvertBetweenNHWCAndNCHW<float>((float*)input_data.data(), tmp_buffer, n, c,
                                                                          h, w, TNN_NS::DataFormatConverter::NHWC2NCHW);
            memcpy(data_raw_buffer.force_to<float*>(), tmp_buffer, count * sizeof(float));
            resource->data = data_raw_buffer;
            delete[] tmp_buffer;
            cur_layer->inputs.erase(cur_layer->inputs.begin());
        } else {
            LOGE("TNN TFLite Gather Converter does not support input dims size %lu\n", input_dims.size());
            return TNN_NS::TNNERR_CONVERT_UNSUPPORT_LAYER;
        }

    } else {
        param->data_in_resource = false;
    }
    const auto& indices_tensor = tf_lite_tensors[tf_lite_operator->inputs[1]];
    const auto& indices_data   = tf_lite_model_buffer[indices_tensor->buffer]->data;
    if (!indices_data.empty()) {
        auto indices_dims = input_tensor->shape;
        if (indices_dims.size() == 2) {
            auto count                           = TNN_NS::DimsVectorUtils::Count(indices_dims);
            TNN_NS::RawBuffer indices_raw_buffer = TNN_NS::RawBuffer(count * sizeof(int32_t), indices_dims);
            indices_raw_buffer.SetDataType(TNN_NS::DATA_TYPE_INT32);
            memcpy(indices_raw_buffer.force_to<float*>(), indices_data.data(), count * sizeof(int32_t));
            resource->indices = indices_raw_buffer;
            cur_layer->inputs.erase(cur_layer->inputs.begin() + 1);
        } else if (indices_dims.size() == 3 || indices_dims.size() == 4) {
            int n, c, h, w;
            if (indices_dims.size() == 3) {
                n = indices_dims[0];
                h = indices_dims[1];
                w = 1;
                c = indices_dims[2];
            } else {
                n = indices_dims[0];
                h = indices_dims[1];
                w = indices_dims[2];
                c = indices_dims[3];
            }
            const int count                      = TNN_NS::DimsVectorUtils::Count(indices_dims);
            TNN_NS::RawBuffer indices_raw_buffer = TNN_NS::RawBuffer(count * sizeof(int32_t), indices_dims);
            indices_raw_buffer.SetDataType(TNN_NS::DATA_TYPE_INT32);
            auto tmp_buffer = new int32_t[count]();
            TNN_NS::DataFormatConverter::ConvertBetweenNHWCAndNCHW<int32_t>(
                (int32_t*)indices_data.data(), tmp_buffer, n, c, h, w, TNN_NS::DataFormatConverter::NHWC2NCHW);
            memcpy(indices_raw_buffer.force_to<float*>(), tmp_buffer, count * sizeof(int32_t));
            resource->indices = indices_raw_buffer;
            delete[] tmp_buffer;
            cur_layer->inputs.erase(cur_layer->inputs.begin() + 1);
        } else {
            LOGE("TNN TFLite Gather convert does not support indices dims size %lu\n", indices_dims.size());
            return TNN_NS::TNNERR_CONVERT_UNSUPPORT_LAYER;
        }
    } else {
        param->indices_in_resource = false;
    }
    return TNN_NS::TNN_CONVERT_OK;
}
using namespace tflite;
REGISTER_CONVERTER(Gather, BuiltinOperator_GATHER);

}  // namespace TNN_CONVERTER