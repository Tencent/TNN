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

#include "tflite-schema/schema_generated.h"
#include "tflite_op_converter.h"
#include "tflite_utils.h"

namespace TNN_CONVERTER {
DECLARE_OP_CONVERTER(Cast);

std::string TFLiteCastConverter::TNNOpType(tflite::BuiltinOperator op_code, bool quantized_model) {
    return "Cast";
}

tflite::ActivationFunctionType TFLiteCastConverter::ActivationType(
    const std::unique_ptr<tflite::OperatorT> &tf_lite_operator, tflite::BuiltinOperator op_code) {
    return tflite::ActivationFunctionType_NONE;
}

TNN_NS::Status TFLiteCastConverter::exec(TNN_NS::NetStructure &net_structure, TNN_NS::NetResource &net_resource,
                                         const std::unique_ptr<tflite::OperatorT> &tf_lite_operator,
                                         const std::vector<std::unique_ptr<tflite::TensorT>> &tf_lite_tensors,
                                         const std::vector<std::unique_ptr<tflite::BufferT>> &tf_lite_model_buffer,
                                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tf_lite_op_set,
                                         bool quantized_model) {
    auto param          = new TNN_NS::CastLayerParam;
    auto cur_layer      = net_structure.layers.back();
    cur_layer->param    = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->name         = cur_layer->name;
    param->type         = cur_layer->type_str;
    param->quantized    = false;
    const auto &options = tf_lite_operator->builtin_options.AsCastOptions();
    if (options == nullptr) {
        param->to = 0;
        return TNN_NS::TNN_CONVERT_OK;
    }
    auto tflite_data_type = options->out_data_type;
    if (tflite_data_type == tflite::TensorType_FLOAT32) {
        param->to = 0;
    } else {
        LOGE("TFLiteCastConverter does not support tflite tensor type\n");
        return TNN_NS::TNNERR_UNSUPPORT_NET;
    }
    return TNN_NS::TNN_CONVERT_OK;
}

using namespace tflite;
REGISTER_CONVERTER(Cast, BuiltinOperator_CAST);

}  // namespace TNN_CONVERTER