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

#include "tf_lite_converter.h"

#include <fstream>
#include <utility>

#include "tflite-schema/schema_generated.h"

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

bool TFLite2Tnn::Convert2Tnn() {
    ReadModel(tf_lite_model_path_);
    const auto& tf_lite_op_set       = tf_lite_model_->operator_codes;
    int sub_graphs_size              = tf_lite_model_->subgraphs.size();
    const auto& tf_lite_model_buffer = tf_lite_model_->buffers;
    bool quantized_mode              = IsQuantized();
    auto& buffer                     = tf_lite_model_->buffers;
    for (int i = 0; i < sub_graphs_size; ++i) {
        const auto& operators = tf_lite_model_->subgraphs[i]->operators;
        const auto& tensors   = tf_lite_model_->subgraphs[i]->tensors;

        // set const
        std::vector<bool> extracted_tensors(tf_lite_model_->subgraphs[i]->tensors.size(), false);

        // set input
        std::vector<std::string> input_list;
        for (const auto index : tf_lite_model_->subgraphs[i]->inputs) {
            const auto& input_tensor         = tensors[index];
            const auto& name                 = input_tensor->name;
            const auto& shape                = input_tensor->shape;

            input_list.push_back(name);

            auto value_info = MakeValueInfo(name, shape, type);
        }

        // set output
        std::vector<std::string> output_list;
        for (const auto index : tf_lite_model_->subgraphs[i]->outputs) {
            const auto& output_tensor        = tensors[index];
            const auto& name                 = output_tensor->name;
            const auto& shape                = output_tensor->shape;

            output_list.push_back(name);

            auto value_info = MakeValueInfo(name, shape, type);
        }

        const int op_nums = operators.size();
        for (int j = 0; j < op_nums; ++j) {
            const int opcode_index = operators[j]->opcode_index;
            const auto op_code     = tf_lite_op_set[opcode_index]->builtin_code;
        }
    }
    return true;
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
        std::cout << "TensorFlow Lite model version ERROR!" << std::endl;
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
            if (opcode == tflite::BuiltinOperator_CONV_2D || opcode == tflite::BuiltinOperator_DEPTHWISE_CONV_2D) {
                const int weight_index    = operators[j]->inputs[1];
                const auto& weight_tensor = tensors[weight_index];
                quantized_mode            = weight_tensor->type == tflite::TensorType_UINT8;
                if (!quantized_mode) {
                    return quantized_mode;
                }
            }
        }
    }
    return quantized_mode;
}

}  // namespace TNN_CONVERTER