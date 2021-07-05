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

namespace TNN_CONVERTER {

DECLARE_OP_CONVERTER(Reduce);

std::string TFLiteReduceConverter::TNNOpType(tflite::BuiltinOperator op_code, bool quantized_model) {
    switch (op_code) {
        case tflite::BuiltinOperator_MEAN:
            return "ReduceMean";
        case tflite::BuiltinOperator_SUM:
            return "ReduceSum";
        default:
            return "";
    }
}
tflite::ActivationFunctionType TFLiteReduceConverter::ActivationType(
    const std::unique_ptr<tflite::OperatorT> &tf_lite_operator, tflite::BuiltinOperator op_code) {
    return tflite::ActivationFunctionType_NONE;
}

TNN_NS::Status TFLiteReduceConverter::exec(TNN_NS::NetStructure &net_structure, TNN_NS::NetResource &net_resource,
                                           const std::unique_ptr<tflite::OperatorT> &tf_lite_operator,
                                           const std::vector<std::unique_ptr<tflite::TensorT>> &tf_lite_tensors,
                                           const std::vector<std::unique_ptr<tflite::BufferT>> &tf_lite_model_buffer,
                                           const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tf_lite_op_set,
                                           bool quantized_model) {
    auto param       = new TNN_NS::ReduceLayerParam;
    auto cur_layer   = net_structure.layers.back();
    cur_layer->param = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->name      = cur_layer->name;
    param->type      = cur_layer->type_str;
    param->quantized = false;
    auto option      = tf_lite_operator->builtin_options.AsReducerOptions();
    param->keep_dims = option->keep_dims;
    assert(cur_layer->inputs.size() == 2);

    auto input_index         = tf_lite_operator->inputs[0];
    const auto &input_tensor = tf_lite_tensors[input_index];
    const auto input_shape   = input_tensor->shape;
    int input_shape_size     = input_shape.size();

    const auto &axes_tensor = tf_lite_tensors[tf_lite_operator->inputs[1]];
    int axes_size           = Count(axes_tensor->shape);
    auto axes_ptr           = reinterpret_cast<int *>(tf_lite_model_buffer[axes_tensor->buffer]->data.data());
    for (int i = 0; i < axes_size; ++i) {
        param->axis.push_back(ConvertAxisFormatTFLite(*axes_ptr, input_shape_size));
        axes_ptr++;
    }
    if (axes_size == 0) {
        param->axis.push_back(ConvertAxisFormatTFLite(*axes_ptr, input_shape_size));
    }

    cur_layer->inputs.resize(1);
    cur_layer->inputs[0] = tf_lite_tensors[tf_lite_operator->inputs[0]]->name;

    return TNN_NS::TNN_CONVERT_OK;
}

using namespace tflite;
REGISTER_CONVERTER(Reduce, BuiltinOperator_MEAN);
REGISTER_CONVERTER(Reduce, BuiltinOperator_SUM);

}  // namespace TNN_CONVERTER
