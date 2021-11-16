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

#include "tflite_converter.h"

#include <fstream>
#include <utility>

#include "tflite-schema/schema_generated.h"
#include "tflite_op_converter.h"
#include "tflite_utils.h"
#include "tnn/core/macro.h"

namespace TNN_CONVERTER {

TFLite2Tnn::TFLite2Tnn(std::string model_path) {
    tf_lite_model_path_ = model_path;
}
TFLite2Tnn::TFLite2Tnn(std::string model_path, std::string onnx_path) {
    // TODO
}
TFLite2Tnn::TFLite2Tnn(std::string mode_path, std::string model_name, std::string onnx_path) {
    tf_lite_model_path_ = mode_path;
    tf_lite_model_name_ = model_name;
    onnx_model_path_    = onnx_path;
}

static bool NeedExtractInput(uint32_t opCode) {
#define NONEED(x)                                                                                                      \
    if (x == opCode)                                                                                                   \
        return false;
    NONEED(tflite::BuiltinOperator_CONV_2D);
    NONEED(tflite::BuiltinOperator_DEPTHWISE_CONV_2D);
    NONEED(tflite::BuiltinOperator_SPLIT);
    NONEED(tflite::BuiltinOperator_CONCATENATION);
    NONEED(tflite::BuiltinOperator_CONV_2D);
    NONEED(tflite::BuiltinOperator_RESHAPE);
    NONEED(tflite::BuiltinOperator_RESIZE_BILINEAR);
    NONEED(tflite::BuiltinOperator_SOFTMAX);

    return true;
}

TNN_NS::Status TFLite2Tnn::Convert2Tnn(TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource) {
    ReadModel(tf_lite_model_path_);
    const auto& tf_lite_op_set       = tf_lite_model_->operator_codes;
    int sub_graphs_size              = tf_lite_model_->subgraphs.size();
    const auto& tf_lite_model_buffer = tf_lite_model_->buffers;
    bool quantized_model             = IsQuantized();
    auto& buffer                     = tf_lite_model_->buffers;
    for (int i = 0; i < sub_graphs_size; ++i) {
        const auto& operators = tf_lite_model_->subgraphs[i]->operators;
        const auto& tensors   = tf_lite_model_->subgraphs[i]->tensors;

        // set const
        std::vector<bool> extracted_tensors(tf_lite_model_->subgraphs[i]->tensors.size(), false);

        // set input
        TNN_NS::InputShapesMap& inputs_shape_map      = net_structure.inputs_shape_map;
        TNN_NS::InputDataTypeMap& input_data_type_map = net_structure.input_data_type_map;
        for (const auto index : tf_lite_model_->subgraphs[i]->inputs) {
            const auto& input_tensor = tensors[index];
            const auto& name         = input_tensor->name;
            std::vector<int32_t> shape(input_tensor->shape);
            ConvertShapeFormatTFLite(shape);
            if (inputs_shape_map.find(name) == inputs_shape_map.end()) {
                inputs_shape_map[name]             = shape;
                const auto& tflite_input_data_type = input_tensor->type;
                input_data_type_map[name]          = GetTnnDataTypeFromTFLite(tflite_input_data_type);
            } else {
                LOGE("The model conflict between same input names %s\n", name.c_str());
                return TNN_NS::TNNERR_CONVERT_INVALID_MODEL;
            }
        }

        // set output
        auto& outputs = net_structure.outputs;
        for (const auto index : tf_lite_model_->subgraphs[i]->outputs) {
            const auto& output_tensor = tensors[index];
            const auto& name          = output_tensor->name;
            std::vector<int32_t> shape(output_tensor->shape);
            ConvertShapeFormatTFLite(shape);
            if (outputs.find(name) == outputs.end()) {
                outputs.insert(name);
            } else {
                LOGE("The model conflict between same output names %s\n", name.c_str());
                return TNN_NS::TNNERR_CONVERT_INVALID_MODEL;
            }
        }
        // convert layer
        auto& layers = net_structure.layers;
        for (int j = 0; j < operators.size(); ++j) {
            const int op_code_index = operators[j]->opcode_index;
            const auto op_code      = tf_lite_op_set[op_code_index]->builtin_code;
            if (NeedExtractInput(op_code)) {
                // TODO
            }
            auto converter = TFLiteOpConverterManager::get()->search(op_code);
            if (converter == nullptr) {
                LOGE("The TFLiteConverter do not support layer:%s\n", tensors[operators[j]->outputs[0]]->name.c_str());
                LOGE("The unsupported operator type is:%s\n",
                     tflite::EnumNameBuiltinOperator(tf_lite_op_set[operators[j]->opcode_index]->builtin_code));
                return TNN_NS::TNNERR_CONVERT_UNSUPPORT_LAYER;
            }
            auto cur_layer = std::make_shared<TNN_NS::LayerInfo>();
            // TNN 默认使用每层op的第一个输出作为层的名称
            cur_layer->name              = tensors[operators[j]->outputs[0]]->name;
            std::string type_name        = converter->TNNOpType(op_code, quantized_model);
            TNN_NS::LayerType layer_type = TNN_NS::GlobalConvertLayerType(type_name);
            cur_layer->type              = layer_type;
            cur_layer->type_str          = type_name;
            for (auto input_index : operators[j]->inputs) {
                if (input_index < 0) {
                    continue;
                }
                cur_layer->inputs.push_back(tensors[input_index]->name);
            }
            for (auto output_index : operators[j]->outputs) {
                cur_layer->outputs.push_back(tensors[output_index]->name);
            }
            net_structure.layers.push_back(cur_layer);
            auto status = converter->exec(net_structure, net_resource, operators[j], tensors, tf_lite_model_buffer,
                                          tf_lite_op_set, quantized_model);
            if (status != TNN_NS::TNN_CONVERT_OK) {
                LOGE("TFLite converter %s failed!\n", cur_layer->type_str.c_str());
                return status;
            }
            tflite::ActivationFunctionType activation_function_type = converter->ActivationType(operators[j], op_code);
            status = converter->SeparateActivation(net_structure, activation_function_type);
            if (status != TNN_NS::TNN_CONVERT_OK) {
                LOGE("TFLite converter %s failed!\n", cur_layer->type_str.c_str());
                return status;
            }
            converter->InsertBlobs(net_structure);
        }
    }
    return TNN_NS::TNN_CONVERT_OK;
}
void TFLite2Tnn::ReadModel(std::string tf_lite_model_path) {
    std::ifstream input_file(tf_lite_model_path, std::ios::binary);
    input_file.seekg(0, std::ios::end);
    const auto file_size = input_file.tellg();
    input_file.seekg(0, std::ios::beg);
    char* buffer = new char[file_size];
    input_file.read(buffer, file_size);
    input_file.close();

    // TODO verify the mode
    flatbuffers::Verifier verify((uint8_t*)buffer, file_size);
    if (!tflite::VerifyModelBuffer(verify)) {
    }

    tf_lite_model_ = tflite::UnPackModel(buffer);
    assert(tf_lite_model_ != nullptr);
    delete[] buffer;
}
bool TFLite2Tnn::IsQuantized() {
    const auto& tf_lite_op_set = tf_lite_model_->operator_codes;
    int sub_graphs_size        = tf_lite_model_->subgraphs.size();
    bool quantized_mode        = true;
    for (int i = 0; i < sub_graphs_size; ++i) {
        const auto& operators   = tf_lite_model_->subgraphs[i]->operators;
        const auto& tensors     = tf_lite_model_->subgraphs[i]->tensors;
        const int operator_size = operators.size();
        for (int j = 0; j < operator_size; ++j) {
            const int opcode_index = operators[j]->opcode_index;
            const auto opcode      = tf_lite_op_set[opcode_index]->builtin_code;
            if (opcode == tflite::BuiltinOperator_CONV_2D || opcode == tflite::BuiltinOperator_DEPTHWISE_CONV_2D ||
                opcode == tflite::BuiltinOperator_FULLY_CONNECTED) {
                const int weight_index    = operators[j]->inputs[1];
                const auto& weight_tensor = tensors[weight_index];
                quantized_mode            = weight_tensor->type == tflite::TensorType_UINT8 || weight_tensor->type == tflite::TensorType_INT8;
                if (!quantized_mode) {
                    return quantized_mode;
                }
            }
        }
    }
    return quantized_mode;
}

}  // namespace TNN_CONVERTER
