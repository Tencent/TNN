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
DECLARE_OP_CONVERTER(Cast);

std::string TFLiteCastConverter::TNNOpType(tflite::BuiltinOperator op_code, bool quantized_model) {
    return "Cast";
}

tflite::ActivationFunctionType TFLiteCastConverter::ActivationType(
    const std::unique_ptr<tflite::OperatorT> &tf_lite_operator, tflite::BuiltinOperator op_code) {
    return tflite::ActivationFunctionType_NONE;
}

static TNN_NS::DataType ConvertDataTypeFromTFLite(tflite::TensorType tf_lite_type) {
    switch (tf_lite_type) {
        case tflite::TensorType_FLOAT32:
        case tflite::TensorType_FLOAT64: {
            return TNN_NS::DATA_TYPE_FLOAT;
        }
        case tflite::TensorType_FLOAT16: {
            return TNN_NS::DATA_TYPE_HALF;
        }
        case tflite::TensorType_BOOL:
        case tflite::TensorType_UINT8:
        case tflite::TensorType_INT8: {
            return TNN_NS::DATA_TYPE_INT8;
        }
        case tflite::TensorType_INT64:
        case tflite::TensorType_INT32: {
            return TNN_NS::DATA_TYPE_INT32;
        }
        default: {
            LOGE("TFLite Converter(Cast) do not support datatype");
            assert(0);
        }
    }
}

TNN_NS::Status TFLiteCastConverter::exec(tnn::NetStructure &net_structure, tnn::NetResource &net_resource,
                                         const std::unique_ptr<tflite::OperatorT> &tf_lite_operator,
                                         const std::vector<std::unique_ptr<tflite::TensorT>> &tf_lite_tensors,
                                         const std::vector<std::unique_ptr<tflite::BufferT>> &tf_lite_model_buffer,
                                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tf_lite_op_set,
                                         bool quantized_model) {
    auto param                = std::make_shared<TNN_NS::CastLayerParam>();
    auto cur_layer            = net_structure.layers.back();
    cur_layer->param          = param;
    param->type               = cur_layer->type_str;
    param->name               = cur_layer->name;
    param->quantized          = false;
    const auto &input_tensor  = tf_lite_tensors[tf_lite_operator->inputs[0]];
    const auto &output_tensor = tf_lite_tensors[tf_lite_operator->outputs[0]];
    param->to                 = ConvertDataTypeFromTFLite(output_tensor->type);
    cur_layer->inputs.resize(1);
    cur_layer->inputs[0] = tf_lite_tensors[tf_lite_operator->inputs[0]]->name;
    return TNN_NS::TNN_CONVERT_OK;
}

using namespace tflite;
REGISTER_CONVERTER(Cast, BuiltinOperator_CAST);
}  // namespace TNN_CONVERTER