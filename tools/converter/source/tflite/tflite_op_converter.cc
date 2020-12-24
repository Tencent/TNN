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

TFLiteOpConverterManager* TFLiteOpConverterManager::tf_lite_op_converter_manager_ = nullptr;

TFLiteOpConverterManager* TFLiteOpConverterManager::get() {
    if (tf_lite_op_converter_manager_ == nullptr) {
        tf_lite_op_converter_manager_ = new TFLiteOpConverterManager;
    }
    return tf_lite_op_converter_manager_;
}
TFLiteOpConverter* TFLiteOpConverterManager::search(const tflite::BuiltinOperator op_index) {
    auto iter = tf_lite_op_converter_map_.find(op_index);
    if (iter == tf_lite_op_converter_map_.end()) {
        return nullptr;
    }
    return iter->second;
}

TFLiteOpConverterManager::~TFLiteOpConverterManager() {
    for (auto& it : tf_lite_op_converter_map_) {
        delete it.second;
    }
    tf_lite_op_converter_map_.clear();
}

void TFLiteOpConverterManager::insert(const tflite::BuiltinOperator op_index, TFLiteOpConverter* t) {
    tf_lite_op_converter_map_.insert(std::make_pair(op_index, t));
}

TNN_NS::Status TFLiteOpConverter::SeparateActivation(TNN_NS::NetStructure& net_structure,
                                                     tflite::ActivationFunctionType activation_function_type) {
    auto& layers = net_structure.layers;
    auto& layer  = layers.back();
    if (activation_function_type == tflite::ActivationFunctionType_NONE || layer->type == TNN_NS::LAYER_CONVOLUTION ||
        layer->type == TNN_NS::LAYER_DECONVOLUTION) {
        return TNN_NS::TNN_CONVERT_OK;
    }
    const std::string conv_output_suffix = "_output";
    const std::string activation_suffix  = "_activation";
    if (activation_function_type == tflite::ActivationFunctionType_RELU ||
        activation_function_type == tflite::ActivationFunctionType_RELU6) {
        auto activation_layer = new TNN_NS::LayerInfo;
        activation_layer->type =
            activation_function_type == tflite::ActivationFunctionType_RELU ? TNN_NS::LAYER_RELU : TNN_NS::LAYER_RELU6;
        activation_layer->type_str = activation_function_type == tflite::ActivationFunctionType_RELU ? "ReLU" : "ReLU6";
        activation_layer->name     = layer->name + activation_suffix;
        activation_layer->inputs.push_back(layer->outputs[0] + conv_output_suffix);
        activation_layer->outputs.push_back(layer->outputs[0]);

        // modify layer
        layer->outputs[0] = layer->outputs[0] + conv_output_suffix;
        // create activation layer
        // create relu param
        auto activation_param       = new TNN_NS::LayerParam;
        activation_layer->param     = std::shared_ptr<TNN_NS::LayerParam>(activation_param);
        activation_param->type      = activation_layer->type_str;
        activation_param->name      = layer->name + activation_suffix;
        activation_param->quantized = false;
        // insert activation layer
        layers.push_back(std::shared_ptr<TNN_NS::LayerInfo>(activation_layer));
    } else {
        LOGE("TNN Converter unsupport activation function\n");
        return TNN_NS::TNNERR_CONVERT_UNSUPPORT_LAYER;
    }
    return TNN_NS::TNN_CONVERT_OK;
}
}  // namespace TNN_CONVERTER
