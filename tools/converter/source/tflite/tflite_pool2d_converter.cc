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
DECLARE_OP_CONVERTER(Pool2D);

std::string TFLitePool2DConverter::TNNOpType(tflite::BuiltinOperator op_code, bool quantized_model) {
    if (quantized_model) {
        return "QuantizedPooling";
    }
    return "Pooling";
}

tflite::ActivationFunctionType TFLitePool2DConverter::ActivationType(
    const std::unique_ptr<tflite::OperatorT>& tf_lite_operator, tflite::BuiltinOperator op_code) {
    return tf_lite_operator->builtin_options.AsPool2DOptions()->fused_activation_function;
}

TNN_NS::Status TFLitePool2DConverter::exec(TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource,
                                           const std::unique_ptr<tflite::OperatorT>& tf_lite_operator,
                                           const std::vector<std::unique_ptr<tflite::TensorT>>& tf_lite_tensors,
                                           const std::vector<std::unique_ptr<tflite::BufferT>>& tf_lite_model_buffer,
                                           const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tf_lite_op_set,
                                           bool quantized_model) {
    TNN_NS::PoolingLayerParam* param = new TNN_NS::PoolingLayerParam;
    auto cur_layer                   = net_structure.layers.back();
    auto tf_lite_op_type             = tf_lite_op_set[tf_lite_operator->opcode_index]->builtin_code;
    const auto& pool_option          = tf_lite_operator->builtin_options.AsPool2DOptions();

    param->name      = cur_layer->name;
    param->type      = cur_layer->type_str;
    param->quantized = quantized_model;

    switch (tf_lite_op_type) {
        case tflite::BuiltinOperator_MAX_POOL_2D: {
            param->pool_type = 0;
            break;
        }
        case tflite::BuiltinOperator_AVERAGE_POOL_2D: {
            param->pool_type = 1;
            break;
        }
        default: {
            LOGE("TNN Pool 2D do not Support unknown pool type\n");
            return TNN_NS::TNNERR_CONVERT_UNSUPPORT_LAYER;
        }
    }

    param->kernels.push_back(pool_option->filter_width);
    param->kernels.push_back(pool_option->filter_height);
    param->kernels_params = param->kernels;

    param->strides.push_back(pool_option->stride_w);
    param->strides.push_back(pool_option->stride_h);

    // default: Padding_SAME
    param->pad_type = 0;
    if (pool_option->padding == tflite::Padding_VALID) {
        // tensorflow pad valid
        param->pad_type = 1;
    }
    param->pads.push_back(0);
    param->pads.push_back(0);
    param->pads.push_back(0);
    param->pads.push_back(0);

    param->kernel_indexs.push_back(-1);
    param->kernel_indexs.push_back(-1);
    // TFLite do not have adaptive pool
    param->is_adaptive_pool = 0;
    param->output_shape     = {-1, -1};
    // update param
    cur_layer->param = std::shared_ptr<TNN_NS::LayerParam>(param);
    if (quantized_model)  {
        // create IntScaleResource for input
        int input_tensor_index = tf_lite_operator->inputs[0];
        TNN_NS::Status status = CreateBlobScaleResource(net_resource, tf_lite_tensors, input_tensor_index);
        if (status != TNN_NS::TNN_CONVERT_OK) {
            return status;
        }
        // create IntScaleResource for output
        int output_tensor_index = tf_lite_operator->outputs[0];
        status = CreateBlobScaleResource(net_resource, tf_lite_tensors, output_tensor_index);
        if (status != TNN_NS::TNN_CONVERT_OK) {
            return status;
        }
    }
    return TNN_NS::TNN_CONVERT_OK;
}

using namespace tflite;
REGISTER_CONVERTER(Pool2D, BuiltinOperator_MAX_POOL_2D);
REGISTER_CONVERTER(Pool2D, BuiltinOperator_AVERAGE_POOL_2D);
}  // namespace TNN_CONVERTER
