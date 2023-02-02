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

namespace TNN_CONVERTER {

DECLARE_OP_CONVERTER(PRelu);

std::string TFLitePReluConverter::TNNOpType(tflite::BuiltinOperator op_code, bool quantized_model) {
    if (quantized_model) {
        // TODO
    }
    return "PReLU";
}

tflite::ActivationFunctionType TFLitePReluConverter::ActivationType(
    const std::unique_ptr<tflite::OperatorT>& tf_lite_operator, tflite::BuiltinOperator op_code) {
    return tflite::ActivationFunctionType_NONE;
}

TNN_NS::Status TFLitePReluConverter::exec(TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource,
                                          const std::unique_ptr<tflite::OperatorT>& tf_lite_operator,
                                          const std::vector<std::unique_ptr<tflite::TensorT>>& tf_lite_tensors,
                                          const std::vector<std::unique_ptr<tflite::BufferT>>& tf_lite_model_buffer,
                                          const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tf_lite_op_set,
                                          bool quantized_model) {
    auto param       = new TNN_NS::PReluLayerParam;
    auto cur_layer   = net_structure.layers.back();
    cur_layer->param = std::shared_ptr<TNN_NS::LayerParam>(param);

    // inputs: input tensor, weight
    const int input_size = tf_lite_operator->inputs.size();
    // weight index
    const int weight_index    = tf_lite_operator->inputs[1];
    const auto& weight_tensor = tf_lite_tensors[weight_index];

    const auto& weight_shape = weight_tensor->shape;
    const int co             = weight_shape[2];
    param->name              = cur_layer->name;
    param->type              = cur_layer->type_str;
    param->quantized         = false;

    auto tf_lite_op_type = tf_lite_op_set[tf_lite_operator->opcode_index]->builtin_code;
    if (tf_lite_op_type == tflite::BuiltinOperator_LEAKY_RELU) {
        param->channel_shared          = 1;
        param->has_filler              = 0;
        auto option                    = tf_lite_operator->builtin_options.AsLeakyReluOptions();
        auto alpha                     = option->alpha;
        auto layer_resource            = new TNN_NS::PReluLayerResource;
        layer_resource->name           = cur_layer->name;
        TNN_NS::RawBuffer slope_handle = TNN_NS::RawBuffer(1 * sizeof(float));
        slope_handle.SetDataType(TNN_NS::DATA_TYPE_FLOAT);
        slope_handle.SetBufferDims({1});
        ::memcpy(slope_handle.force_to<float*>(), &alpha, 1 * sizeof(float));
        layer_resource->slope_handle               = slope_handle;
        net_resource.resource_map[cur_layer->name] = std::shared_ptr<TNN_NS::LayerResource>(layer_resource);
    } else if (tf_lite_op_type == tflite::BuiltinOperator_PRELU) {
        ASSERT(input_size == 2);
        param->channel_shared          = 0;
        param->has_filler              = 0;
        auto layer_resource            = new TNN_NS::PReluLayerResource;
        layer_resource->name           = cur_layer->name;
        TNN_NS::RawBuffer slope_handle = TNN_NS::RawBuffer(co * sizeof(float));
        slope_handle.SetDataType(TNN_NS::DATA_TYPE_FLOAT);
        slope_handle.SetBufferDims({co});
        auto data_ptr = reinterpret_cast<const float*>(tf_lite_model_buffer[weight_tensor->buffer]->data.data());
        ::memcpy(slope_handle.force_to<float*>(), data_ptr, sizeof(float) * co);
        layer_resource->slope_handle = slope_handle;

        net_resource.resource_map[cur_layer->name] = std::shared_ptr<TNN_NS::LayerResource>(layer_resource);
    }

    cur_layer->inputs.resize(1);
    cur_layer->inputs[0] = tf_lite_tensors[tf_lite_operator->inputs[0]]->name;

    return TNN_NS::TNN_CONVERT_OK;
}
using namespace tflite;
REGISTER_CONVERTER(PRelu, BuiltinOperator_PRELU);
REGISTER_CONVERTER(PRelu, BuiltinOperator_LEAKY_RELU);
}  // namespace TNN_CONVERTER
