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

#include "tflite_op_converter.h"
#include "tflite_utils.h"
#include "tnn//utils/dims_vector_utils.h"
#include "tnn/utils/data_format_converter.h"

namespace TNN_CONVERTER {

DECLARE_OP_CONVERTER(FullyConnected);

std::string TFLiteFullyConnectedConverter::TNNOpType(tflite::BuiltinOperator op_code, bool quantized_model) {
    if (quantized_model) {
        return "QuantizedInnerProduct";
    }
    return "InnerProduct";
};

tflite::ActivationFunctionType TFLiteFullyConnectedConverter::ActivationType(
    const std::unique_ptr<tflite::OperatorT> &tf_lite_operator, tflite::BuiltinOperator op_code) {
    return tf_lite_operator->builtin_options.AsFullyConnectedOptions()->fused_activation_function;
}

static TNN_NS::Status CreateResource(TNN_NS::NetResource &net_resource,
                                     const std::unique_ptr<tflite::OperatorT> &tf_lite_operator,
                                     const std::vector<std::unique_ptr<tflite::TensorT>> &tf_lite_tensors,
                                     const std::vector<std::unique_ptr<tflite::BufferT>> &tf_lite_model_buffer,
                                     std::string &layer_name) {
    if (net_resource.resource_map.find(layer_name) == net_resource.resource_map.end()) {
        return TNN_NS::Status(TNN_NS::TNNERR_CONVERT_INVALID_MODEL, "Create InnerProduct layer failed\n");
    }
    auto resource = reinterpret_cast<TNN_NS::InnerProductLayerResource *>(net_resource.resource_map[layer_name].get());
    auto &input_tensor  = tf_lite_tensors[tf_lite_operator->inputs[0]];
    auto &input_dims    = input_tensor->shape;
    auto &weight_tensor = tf_lite_tensors[tf_lite_operator->inputs[1]];
    auto &weight_dims   = weight_tensor->shape;
    int weight_count    = Count(weight_dims);
    if (input_dims.size() == 4) {
        // create weight
        int n              = input_dims[0];
        int h              = input_dims[1];
        int w              = input_dims[2];
        int c              = input_dims[3];
        int output_channel = weight_dims[0];
        int feature_size   = weight_dims[1];
        auto tensor_data   = static_cast<uint8_t *>(tf_lite_model_buffer[weight_tensor->buffer]->data.data());
        auto tmp_buffer    = new uint8_t[weight_count]();
        for (int i = 0; i < output_channel; ++i) {
            auto data_ptr = &tensor_data[i * feature_size];
            TNN_NS::DataFormatConverter::ConvertBetweenNHWCAndNCHW<uint8_t>(
                data_ptr, &tmp_buffer[i * feature_size], n, c, h, w, TNN_NS::DataFormatConverter::NHWC2NCHW);
        }
        auto weight_handle = TNN_NS::RawBuffer(weight_count * sizeof(int8_t));
        weight_handle.SetDataType(TNN_NS::DATA_TYPE_INT8);
        auto &weight_zero_point = weight_tensor->quantization->zero_point;
        ASSERT(weight_zero_point.size() == 1);
        auto weight_handle_data = weight_handle.force_to<int8_t *>();
        for (int i = 0; i < weight_count; ++i) {
            weight_handle_data[i] = (int8_t)(tmp_buffer[i] - weight_zero_point[0]);
        }
        resource->weight_handle = weight_handle;
        delete[] tmp_buffer;
        // create weight scale
        auto &input_tensor = tf_lite_tensors[tf_lite_operator->inputs[0]];
        auto &input_scale  = input_tensor->quantization->scale;
        auto &weight_scale = weight_tensor->quantization->scale;
        ASSERT(input_scale.size() == weight_scale.size());
        auto scale_handle = TNN_NS::RawBuffer(weight_scale.size() * sizeof(float));
        scale_handle.SetDataType(TNN_NS::DATA_TYPE_FLOAT);
        auto scale_data = scale_handle.force_to<float *>();
        for (int i = 0; i < weight_scale.size(); ++i) {
            scale_data[i] = input_scale[i] * weight_scale[i];
        }
        resource->scale_handle = scale_handle;
        // for symmetric quantization zero point always is 0
        auto zero_point_handle = TNN_NS::RawBuffer(weight_scale.size() * sizeof(int8_t));
        zero_point_handle.SetDataType(TNN_NS::DATA_TYPE_INT8);
        resource->zero_point_handle = zero_point_handle;
    } else {
        return TNN_NS::Status(TNN_NS::TNNERR_CONVERT_UNSUPPORT_LAYER, "Quantized TFLite Converter do not support\n");
    }
    if (tf_lite_operator->inputs.size() == 3) {
        // bias
        auto &bias_tensor             = tf_lite_tensors[tf_lite_operator->inputs[2]];
        auto bias_ptr                 = (int32_t *)(tf_lite_model_buffer[bias_tensor->buffer]->data.data());
        auto bias_dims                = bias_tensor->shape;
        int bias_count                = Count(bias_dims);
        TNN_NS::RawBuffer bias_handle = TNN_NS::RawBuffer(bias_count * sizeof(int32_t));
        ::memcpy(bias_handle.force_to<int32_t *>(), bias_ptr, bias_count * sizeof(int32_t));
        resource->bias_handle = bias_handle;
    }
    return TNN_NS::TNN_CONVERT_OK;
}

TNN_NS::Status TFLiteFullyConnectedConverter::exec(
    TNN_NS::NetStructure &net_structure, TNN_NS::NetResource &net_resource,
    const std::unique_ptr<tflite::OperatorT> &tf_lite_operator,
    const std::vector<std::unique_ptr<tflite::TensorT>> &tf_lite_tensors,
    const std::vector<std::unique_ptr<tflite::BufferT>> &tf_lite_model_buffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tf_lite_op_set, bool quantized_model) {
    TNN_NS::InnerProductLayerParam *param = new TNN_NS::InnerProductLayerParam;
    auto cur_layer                        = net_structure.layers.back();
    cur_layer->param                      = std::shared_ptr<TNN_NS::LayerParam>(param);
    assert(tf_lite_operator->inputs.size() == 2 || tf_lite_operator->inputs.size() == 3);
    auto &weight_tensor = tf_lite_tensors[tf_lite_operator->inputs[1]];
    auto weight_shape   = weight_tensor->shape;
    assert(weight_shape.size() == 2);
    param->type       = cur_layer->type_str;
    param->name       = cur_layer->name;
    param->quantized  = quantized_model;
    param->axis       = 1;
    param->transpose  = 0;
    const int co      = weight_shape[0];
    param->num_output = co;
    if (tf_lite_operator->inputs.size() == 3) {
        param->has_bias = 1;
    }
    auto layer_resource                        = std::make_shared<TNN_NS::InnerProductLayerResource>();
    layer_resource->name                       = cur_layer->name;
    net_resource.resource_map[cur_layer->name] = layer_resource;
    int weight_size                            = Count(weight_shape);
    assert(weight_size > 0);
    if (quantized_model) {
        // create IntScaleResource for input
        int input_tensor_index = tf_lite_operator->inputs[0];
        auto status            = CreateBlobScaleResource(net_resource, tf_lite_tensors, input_tensor_index);
        ASSERT(status == TNN_NS::TNN_CONVERT_OK);
        // crate InnerProduct layer resource
        status = CreateResource(net_resource, tf_lite_operator, tf_lite_tensors, tf_lite_model_buffer, cur_layer->name);
        ASSERT(status == TNN_NS::TNN_CONVERT_OK);
        // create IntScaleResource for output
        int output_tensor_index = tf_lite_operator->outputs[0];
        status                  = CreateBlobScaleResource(net_resource, tf_lite_tensors, output_tensor_index);
        ASSERT(status == TNN_NS::TNN_CONVERT_OK);
    } else {

        auto weight_ptr       = reinterpret_cast<float *>(tf_lite_model_buffer[weight_tensor->buffer]->data.data());
        auto input_data_shape = tf_lite_tensors[tf_lite_operator->inputs[0]]->shape;
        if (input_data_shape.size() == 4) {
            int n            = input_data_shape[0];
            int h            = input_data_shape[1];
            int w            = input_data_shape[2];
            int c            = input_data_shape[3];
            int feature_size = weight_shape[1];
            auto tmp         = new float[weight_size]();
            for (int i = 0; i < co; ++i) {
                auto data_ptr = weight_ptr + (i * feature_size);
                TNN_NS::DataFormatConverter::ConvertBetweenNHWCAndNCHW<float>(
                    data_ptr, &tmp[i * feature_size], n, c, h, w, TNN_NS::DataFormatConverter::NHWC2NCHW);
            }
            TNN_NS::RawBuffer weight_handle = TNN_NS::RawBuffer(weight_size * sizeof(float));
            ::memcpy(weight_handle.force_to<float *>(), tmp, weight_size * sizeof(float));
            layer_resource->weight_handle = ConvertRawBuffer::GetInstance()->Convert(weight_handle);
            delete[] tmp;
        } else {
            TNN_NS::RawBuffer weight_handle = TNN_NS::RawBuffer(weight_size * sizeof(float));
            ::memcpy(weight_handle.force_to<float *>(), weight_ptr, weight_size * sizeof(float));
            layer_resource->weight_handle = ConvertRawBuffer::GetInstance()->Convert(weight_handle);
        }
        if (tf_lite_operator->inputs.size() == 3 && tf_lite_operator->inputs[2] >= 0) {
            auto &bias_tensor = tf_lite_tensors[tf_lite_operator->inputs[2]];
            auto bias_ptr     = reinterpret_cast<const float *>(tf_lite_model_buffer[bias_tensor->buffer]->data.data());
            auto bias_shape   = bias_tensor->shape;
            int bias_size     = Count(bias_shape);
            TNN_NS::RawBuffer bias_handle = TNN_NS::RawBuffer(bias_size * sizeof(float));
            ::memcpy(bias_handle.force_to<float *>(), bias_ptr, bias_size * sizeof(float));
            layer_resource->bias_handle = ConvertRawBuffer::GetInstance()->Convert(bias_handle);
        }
    }

    cur_layer->inputs.resize(1);
    cur_layer->outputs.resize(1);
    cur_layer->inputs[0]  = tf_lite_tensors[tf_lite_operator->inputs[0]]->name;
    cur_layer->outputs[0] = tf_lite_tensors[tf_lite_operator->outputs[0]]->name;
    return TNN_NS::TNN_CONVERT_OK;
}

using namespace tflite;
REGISTER_CONVERTER(FullyConnected, BuiltinOperator_FULLY_CONNECTED);

}  // namespace TNN_CONVERTER
