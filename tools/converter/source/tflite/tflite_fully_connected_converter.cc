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
#include "tnn/utils/data_format_converter.h"

namespace TNN_CONVERTER {

DECLARE_OP_CONVERTER(FullyConnected);

std::string TFLiteFullyConnectedConverter::TNNOpType(tflite::BuiltinOperator op_code, bool quantized_model) {
    return "InnerProduct";
};

tflite::ActivationFunctionType TFLiteFullyConnectedConverter::ActivationType(
    const std::unique_ptr<tflite::OperatorT> &tf_lite_operator, tflite::BuiltinOperator op_code) {
    return tf_lite_operator->builtin_options.AsFullyConnectedOptions()->fused_activation_function;
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
    param->quantized  = false;
    param->axis       = 1;
    param->transpose  = 1;
    const int co      = weight_shape[0];
    param->num_output = co;
    if (quantized_model) {
        // TODO
    } else {
        auto layer_resource  = new TNN_NS::InnerProductLayerResource;
        layer_resource->name = cur_layer->name;
        int weight_size      = Count(weight_shape);
        assert(weight_size > 0);
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
            layer_resource->weight_handle = weight_handle;
            delete[] tmp;
        } else {
            TNN_NS::RawBuffer weight_handle = TNN_NS::RawBuffer(weight_size * sizeof(float));
            ::memcpy(weight_handle.force_to<float *>(), weight_ptr, weight_size * sizeof(float));
            layer_resource->weight_handle = weight_handle;
        }
        if (tf_lite_operator->inputs.size() == 3) {
            param->has_bias   = 1;
            auto &bias_tensor = tf_lite_tensors[tf_lite_operator->inputs[2]];
            auto bias_ptr     = reinterpret_cast<const float *>(tf_lite_model_buffer[bias_tensor->buffer]->data.data());
            auto bias_shape   = bias_tensor->shape;
            int bias_size     = Count(bias_shape);
            TNN_NS::RawBuffer bias_handle = TNN_NS::RawBuffer(bias_size * sizeof(float));
            ::memcpy(bias_handle.force_to<float *>(), bias_ptr, bias_size * sizeof(float));
            layer_resource->bias_handle = bias_handle;
        }
        net_resource.resource_map[cur_layer->name] = std::shared_ptr<TNN_NS::LayerResource>(layer_resource);
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
