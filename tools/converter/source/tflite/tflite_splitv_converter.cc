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
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_CONVERTER {
DECLARE_OP_CONVERTER(SplitV);

std::string TFLiteSplitVConverter::TNNOpType(tflite::BuiltinOperator op_code, bool quantized_model) {
    return "SplitV";
}

tflite::ActivationFunctionType TFLiteSplitVConverter::ActivationType(
    const std::unique_ptr<tflite::OperatorT> &tf_lite_operator, tflite::BuiltinOperator op_code) {
    return tflite::ActivationFunctionType_NONE;
}

TNN_NS::Status TFLiteSplitVConverter::exec(TNN_NS::NetStructure &net_structure, TNN_NS::NetResource &net_resource,
                                           const std::unique_ptr<tflite::OperatorT> &tf_lite_operator,
                                           const std::vector<std::unique_ptr<tflite::TensorT>> &tf_lite_tensors,
                                           const std::vector<std::unique_ptr<tflite::BufferT>> &tf_lite_model_buffer,
                                           const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tf_lite_op_set,
                                           bool quantized_model) {
    auto parm            = new TNN_NS::SplitVLayerParam;
    auto cur_layer       = net_structure.layers.back();
    cur_layer->param     = std::shared_ptr<TNN_NS::LayerParam>(parm);
    parm->type           = cur_layer->type_str;
    parm->name           = cur_layer->name;
    parm->quantized      = false;
    auto tf_lite_op_type = tf_lite_op_set[tf_lite_operator->opcode_index]->builtin_code;
    if (tf_lite_op_type == tflite::BuiltinOperator_SPLIT_V) {
        auto option     = tf_lite_operator->builtin_options.AsSplitVOptions();
        auto num_splits = option->num_splits;
        ASSERT(tf_lite_operator->inputs.size() >= 2);
        auto &splits_tensor = tf_lite_tensors[tf_lite_operator->inputs[1]];
        auto data_size =
            tf_lite_model_buffer[splits_tensor->buffer]->data.size() / SizeofTFLiteTensorData(splits_tensor->type);
        ASSERT(data_size == num_splits);
        auto data_ptr = reinterpret_cast<int32_t *>(tf_lite_model_buffer[splits_tensor->buffer]->data.data());
        int sum       = 0;
        for (int i = 0; i < data_size; ++i) {
            sum += data_ptr[i];
            parm->slices.push_back(data_ptr[i]);
        }
        if (tf_lite_operator->inputs.size() == 3) {
            auto &axis_tensor = tf_lite_tensors[tf_lite_operator->inputs[2]];
            auto axis_size =
                tf_lite_model_buffer[axis_tensor->buffer]->data.size() / SizeofTFLiteTensorData(axis_tensor->type);
            ASSERT(axis_size == 1);
            auto &value_tensor = tf_lite_tensors[tf_lite_operator->inputs[0]];
            int axis           = *(reinterpret_cast<int32_t *>(tf_lite_model_buffer[axis_tensor->buffer]->data.data()));
            if (axis < 0) {
                axis += value_tensor->shape.size();
            }
            parm->axis = ConvertAxisFormatTFLite(axis);
            ASSERT(sum == value_tensor->shape[axis]);

        } else {
            parm->axis = 0;
        }
        cur_layer->inputs.resize(1);
        cur_layer->inputs[0] = tf_lite_tensors[tf_lite_operator->inputs[0]]->name;
        return TNN_NS::TNN_CONVERT_OK;
    } else if (tf_lite_op_type == tflite::BuiltinOperator_SPLIT) {
        auto option     = tf_lite_operator->builtin_options.AsSplitOptions();
        auto num_splits = option->num_splits;
        ASSERT(num_splits == tf_lite_operator->outputs.size());
        ASSERT(tf_lite_operator->inputs.size() >= 2);
        auto &axis_tensor  = tf_lite_tensors[tf_lite_operator->inputs[0]];
        auto &input_tensor = tf_lite_tensors[tf_lite_operator->inputs[1]];
        auto axis_size =
            tf_lite_model_buffer[axis_tensor->buffer]->data.size() / SizeofTFLiteTensorData(axis_tensor->type);
        ASSERT(axis_size == 1);
        int axis = *(reinterpret_cast<int32_t *>(tf_lite_model_buffer[axis_tensor->buffer]->data.data()));
        if (axis < 0) {
            axis += input_tensor->shape.size();
        }
        parm->axis = ConvertAxisFormatTFLite(axis);
        cur_layer->inputs.resize(1);
        cur_layer->inputs[0] = tf_lite_tensors[tf_lite_operator->inputs[1]]->name;
        return TNN_NS::TNN_CONVERT_OK;
    }
    return TNN_NS::TNNERR_CONVERT_UNSUPPORT_LAYER;
}

using namespace tflite;
REGISTER_CONVERTER(SplitV, BuiltinOperator_SPLIT_V);
REGISTER_CONVERTER(SplitV, BuiltinOperator_SPLIT);
}  // namespace TNN_CONVERTER
