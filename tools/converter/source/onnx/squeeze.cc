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

DECLARE_OP_COVERTER(SqueezeTFLite);

std::string SqueezeTFLite::op_type() {
    return "Squeeze";
}

void SqueezeTFLite::run(NodeInfo& dst_op, const std::unique_ptr<tflite::OperatorT>& tf_lite_op,
                        const std::vector<std::unique_ptr<tflite::TensorT>>& tf_lite_tensors,
                        const std::vector<std::unique_ptr<tflite::BufferT>>& tf_lite_model_buffer,
                        const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tf_lite_op_set) {
    const auto& tf_lite_squeeze_option = tf_lite_op->builtin_options.AsSqueezeOptions();
    auto squeeze_dims                  = tf_lite_squeeze_option->squeezeDims;

    // nhwc -> nchw
    for (int i = 0; i < squeeze_dims.size(); ++i) {
        if (squeeze_dims[i] == 1) {
            squeeze_dims[i] = 3;
        } else if (squeeze_dims[i] == 3) {
            squeeze_dims[i] = 1;
        }
    }

    onnx::NodeProto squeeze;
    const auto input_index         = tf_lite_op->inputs[0];
    const auto input_name          = tf_lite_tensors[input_index]->name;
    const auto output_index        = tf_lite_op->outputs[0];
    const auto output_name         = tf_lite_tensors[output_index]->name;
    onnx::AttributeProto axes_attr = MakeAttribute("axes", squeeze_dims);

    squeeze.set_op_type("Squeeze");
    squeeze.add_input(input_name);
    squeeze.add_output(output_name);
    squeeze.add_attribute()->CopyFrom(squeeze);

    dst_op.node_proto = std::move(squeeze);
}

using namespace tflite;
REGISTER_CONVERTER(SqueezeTFLite, BuiltinOperator_SQUEEZE);
