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

DECLARE_OP_CONVERTER(DepthToSpace);

std::string TFLiteDepthToSpaceConverter::TNNOpType(tflite::BuiltinOperator op_code, bool quantized_model) {
    if (quantized_model) {
        // TODO
    }
    return "Reorg";
}

tflite::ActivationFunctionType TFLiteDepthToSpaceConverter::ActivationType(
    const std::unique_ptr<tflite::OperatorT>& tf_lite_operator, tflite::BuiltinOperator op_code) {
    return tflite::ActivationFunctionType_NONE;
}

TNN_NS::Status TFLiteDepthToSpaceConverter::exec(
    TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource,
    const std::unique_ptr<tflite::OperatorT>& tf_lite_operator,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tf_lite_tensors,
    const std::vector<std::unique_ptr<tflite::BufferT>>& tf_lite_model_buffer,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tf_lite_op_set, bool quantized_model) {
    auto param           = new TNN_NS::ReorgLayerParam;
    auto cur_layer       = net_structure.layers.back();
    cur_layer->param     = std::shared_ptr<TNN_NS::LayerParam>(param);
    auto tf_lite_op_type = tf_lite_op_set[tf_lite_operator->opcode_index]->builtin_code;

    param->name      = cur_layer->name;
    param->type      = cur_layer->type_str;
    param->quantized = false;

    param->mode = 0;
    switch (tf_lite_op_type) {
        case tflite::BuiltinOperator_DEPTH_TO_SPACE:
            param->forward = true;
            param->stride  = tf_lite_operator->builtin_options.AsDepthToSpaceOptions()->block_size;
            break;
        case tflite::BuiltinOperator_SPACE_TO_DEPTH:
            param->forward = false;
            param->stride  = tf_lite_operator->builtin_options.AsSpaceToDepthOptions()->block_size;
            break;
        default:
            LOGE("TNN Reorg unknown op type\n");
            return TNN_NS::TNNERR_CONVERT_UNSUPPORT_LAYER;
    }

    return TNN_NS::TNN_CONVERT_OK;
}
using namespace tflite;
REGISTER_CONVERTER(DepthToSpace, BuiltinOperator_DEPTH_TO_SPACE);
REGISTER_CONVERTER(DepthToSpace, BuiltinOperator_SPACE_TO_DEPTH);
}  // namespace TNN_CONVERTER
