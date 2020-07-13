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

#include "tf_lite_op_converter.h"

DECLARE_OP_COVERTER(TransposeTFLite);

std::string TransposeTFLite::op_type() {
    return "Transpose";
}

void TransposeTFLite::run(NodeInfo& dst_op, const std::unique_ptr<tflite::OperatorT>& tf_lite_op,
                          const std::vector<std::unique_ptr<tflite::TensorT>>& tf_lite_tensors,
                          const std::vector<std::unique_ptr<tflite::BufferT>>& tf_lite_model_buffer,
                          const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tf_lite_op_set) {
    const auto& tf_lite_transpose_option = tf_lite_op->builtin_options.AsTransposeOptions();
}

using namespace tflite;
REGISTER_CONVERTER(TransposeTFLite, BuiltinOperator_TRANSPOSE);
