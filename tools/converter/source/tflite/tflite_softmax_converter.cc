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

DECLARE_OP_CONVERTER(Softmax);

std::string TFLiteSoftmaxConverter::TNNOpType(bool quantizedModel) {
    if (quantizedModel) {
        return "QuantizedSoftmax";
    }
    return "Softmax";
}

TNN_NS::Status TFLiteSoftmaxConverter::exec(TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource,
                                            const std::unique_ptr<tflite::OperatorT>& tf_lite_operator,
                                            const std::vector<std::unique_ptr<tflite::TensorT>>& tf_lite_tensors,
                                            const std::vector<std::unique_ptr<tflite::BufferT>>& tf_lite_model_buffer,
                                            const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tf_lite_op_set,
                                            bool quantizedModel) {
    ASSERT(tf_lite_operator->inputs.size() == 1);

    TNN_NS::SoftmaxLayerParam* param = new TNN_NS::SoftmaxLayerParam;
    auto cur_layer                   = net_structure.layers.back();
    auto tf_lite_op_type             = tf_lite_op_set[tf_lite_operator->opcode_index]->builtin_code;
    const auto& softmax_option       = tf_lite_operator->builtin_options.AsSoftmaxOptions();

    if (quantizedModel) {
        // TODO
    } else {
        param->name      = cur_layer->name;
        param->type      = cur_layer->type_str;
        param->quantized = false;
        param->axis      = 1;

        // update param
        cur_layer->param = std::shared_ptr<TNN_NS::LayerParam>(param);
    }

    // set input output index
    cur_layer->inputs.resize(1);
    cur_layer->outputs.resize(1);
    cur_layer->inputs[0]  = tf_lite_tensors[tf_lite_operator->inputs[0]]->name;
    cur_layer->outputs[0] = tf_lite_tensors[tf_lite_operator->outputs[0]]->name;
    return TNN_NS::TNN_CONVERT_OK;
}

using namespace tflite;
REGISTER_CONVERTER(Softmax, BuiltinOperator_SOFTMAX);
}  // namespace TNN_CONVERTER
