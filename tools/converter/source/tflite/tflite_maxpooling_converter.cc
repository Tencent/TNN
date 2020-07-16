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
DECLARE_OP_CONVERTER(Pooling);

std::string TFLitePoolingConverter::TNNOpType(bool quantizedModel) {
    if (quantizedModel) {
        return "QuantizedPooling";
    }
    return "Pooling";
}

TNN_NS::Status TFLitePoolingConverter::exec(
    TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource,
    const std::unique_ptr<tflite::OperatorT>& tf_lite_operator,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tf_lite_tensors,
    const std::vector<std::unique_ptr<tflite::BufferT>>& tf_lite_model_buffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tf_lite_op_set, bool quantizedModel) {
    TNN_NS::PoolingLayerParam* param = new TNN_NS::PoolingLayerParam;
    auto cur_layer                = net_structure.layers.back();
    auto tf_lite_op_type          = tf_lite_op_set[tf_lite_operator->opcode_index]->builtin_code;
    const auto& pool_option = tf_lite_operator->builtin_options.AsPool2DOptions();

    if (quantizedModel) {
        // TODO
    } else {
        ASSERT(pool_option->fused_activation_function == tflite::ActivationFunctionType_NONE);

        param->name = cur_layer->name;
        param->type = cur_layer->type_str;
        param->quantized = false;
        param->kernels.push_back(pool_option->filter_width);
        param->kernels.push_back(pool_option->filter_height);

        param->kernels_params = param->kernels;

        param->strides.push_back(pool_option->stride_w);
        param->strides.push_back(pool_option->stride_h);

        param->pad_type = 0;
        if (pool_option->padding == tflite::Padding_VALID) {
            // tensorflow pad valid
            param->pad_type = 1;
            param->pads.push_back(0);
            param->pads.push_back(0);
            param->pads.push_back(0);
            param->pads.push_back(0);
        } else if (pool_option->padding == tflite::Padding_SAME) {
            param->pad_type = 0;
            param->pads.push_back(0);
            param->pads.push_back(0);
            param->pads.push_back(0);
            param->pads.push_back(0);
        }

        param->pool_type = 1;
        const auto op_index = tf_lite_operator->opcode_index;
        auto op_type = tf_lite_op_set[op_index]->builtin_code;
        if (op_type == tflite::BuiltinOperator_MAX_POOL_2D) {
            param->pool_type = 0;
        }

        param->kernel_indexs.push_back(0);
        param->kernel_indexs.push_back(0);

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
REGISTER_CONVERTER(Pooling, BuiltinOperator_MAX_POOL_2D);
}
