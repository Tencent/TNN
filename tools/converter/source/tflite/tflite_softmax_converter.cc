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

std::string TFLiteSoftmaxConverter::TNNOpType(tflite::BuiltinOperator op_code, bool quantized_model) {
    if (quantized_model) {
        return "QuantizedSoftmax";
    }
    return "Softmax";
}

tflite::ActivationFunctionType TFLiteSoftmaxConverter::ActivationType(
    const std::unique_ptr<tflite::OperatorT>& tf_lite_operator, tflite::BuiltinOperator op_code) {
    return tflite::ActivationFunctionType_NONE;
}

TNN_NS::Status TFLiteSoftmaxConverter::exec(TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource,
                                            const std::unique_ptr<tflite::OperatorT>& tf_lite_operator,
                                            const std::vector<std::unique_ptr<tflite::TensorT>>& tf_lite_tensors,
                                            const std::vector<std::unique_ptr<tflite::BufferT>>& tf_lite_model_buffer,
                                            const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tf_lite_op_set,
                                            bool quantized_model) {
    ASSERT(tf_lite_operator->inputs.size() == 1);

    TNN_NS::SoftmaxLayerParam* param = new TNN_NS::SoftmaxLayerParam;
    auto cur_layer                   = net_structure.layers.back();
    cur_layer->param                 = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->name                      = cur_layer->name;
    param->type                      = cur_layer->type_str;
    param->quantized                 = false;
    param->axis                      = 1;
    return TNN_NS::TNN_CONVERT_OK;
}

using namespace tflite;
REGISTER_CONVERTER(Softmax, BuiltinOperator_SOFTMAX);
}  // namespace TNN_CONVERTER
