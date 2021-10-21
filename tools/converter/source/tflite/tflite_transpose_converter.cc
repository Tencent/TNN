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
DECLARE_OP_CONVERTER(Transpose);

std::string TFLiteTransposeConverter::TNNOpType(tflite::BuiltinOperator op_code, bool quantized_model) {
    if (quantized_model) {
        return "QuantizedPermute";
    }
    return "Permute";
}
tflite::ActivationFunctionType TFLiteTransposeConverter::ActivationType(
    const std::unique_ptr<tflite::OperatorT>& tf_lite_operator, tflite::BuiltinOperator op_code) {
    return tflite::ActivationFunctionType_NONE;
}

TNN_NS::Status TFLiteTransposeConverter::exec(TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource,
                                              const std::unique_ptr<tflite::OperatorT>& tf_lite_operator,
                                              const std::vector<std::unique_ptr<tflite::TensorT>>& tf_lite_tensors,
                                              const std::vector<std::unique_ptr<tflite::BufferT>>& tf_lite_model_buffer,
                                              const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tf_lite_op_set,
                                              bool quantized_model) {
    TNN_NS::PermuteLayerParam* param = new TNN_NS::PermuteLayerParam;
    auto cur_layer                   = net_structure.layers.back();
    cur_layer->param                 = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->name                      = cur_layer->name;
    param->type                      = cur_layer->type_str;
    param->quantized                 = quantized_model;

    std::vector<int> perm;
    if (tf_lite_operator->inputs.size() == 2) {
        const auto& shape_tensor = tf_lite_tensors[tf_lite_operator->inputs[1]];
        assert(shape_tensor->type == tflite::TensorType_INT32);
        int shape_size = 1;
        for (int i = 0; i < shape_tensor->shape.size(); ++i) {
            shape_size *= shape_tensor->shape[i];
        }
        const auto& shape_data = tf_lite_model_buffer[shape_tensor->buffer]->data;
        ASSERT(shape_size == shape_data.size() / 4);

        auto shape_data_ptr = reinterpret_cast<const int32_t*>(shape_data.data());
        perm.assign(shape_data_ptr, shape_data_ptr + shape_size);
    } else {
        LOGE("TNN Transpose do not support type\n");
        return TNN_NS::TNNERR_CONVERT_UNSUPPORT_LAYER;
    }
    auto status = ConvertPermFormatTFLite(perm);
    if (!status) {
        return TNN_NS::TNNERR_CONVERT_UNSUPPORT_LAYER;
    }
    param->orders = perm;
    if (quantized_model) {
       // create input blob scale
       auto input_index = tf_lite_operator->inputs[0];
       auto status      = CreateBlobScaleResource(net_resource, tf_lite_tensors, input_index);
       ASSERT(status == TNN_NS::TNN_CONVERT_OK);
       // creat output blob scale
       auto output_index = tf_lite_operator->outputs[0];
       status            = CreateBlobScaleResource(net_resource, tf_lite_tensors, output_index);
       ASSERT(status == TNN_NS::TNN_CONVERT_OK);
    }

    // set input output index
    cur_layer->inputs.resize(1);
    cur_layer->inputs[0] = tf_lite_tensors[tf_lite_operator->inputs[0]]->name;
    return TNN_NS::TNN_CONVERT_OK;
}

using namespace tflite;
REGISTER_CONVERTER(Transpose, BuiltinOperator_TRANSPOSE);
}  // namespace TNN_CONVERTER
