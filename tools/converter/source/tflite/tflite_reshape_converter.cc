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
DECLARE_OP_CONVERTER(Reshape);

std::string TFLiteReshapeConverter::TNNOpType(tflite::BuiltinOperator op_code, bool quantized_model) {
    if (quantized_model) {
        return "QuantizedReshape";
    }
    return "Reshape";
}
tflite::ActivationFunctionType TFLiteReshapeConverter::ActivationType(
    const std::unique_ptr<tflite::OperatorT>& tf_lite_operator, tflite::BuiltinOperator op_code) {
    return tflite::ActivationFunctionType_NONE;
}

TNN_NS::Status TFLiteReshapeConverter::exec(TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource,
                                            const std::unique_ptr<tflite::OperatorT>& tf_lite_operator,
                                            const std::vector<std::unique_ptr<tflite::TensorT>>& tf_lite_tensors,
                                            const std::vector<std::unique_ptr<tflite::BufferT>>& tf_lite_model_buffer,
                                            const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tf_lite_op_set,
                                            bool quantized_model) {
    TNN_NS::ReshapeLayerParam* param = new TNN_NS::ReshapeLayerParam;
    auto cur_layer                   = net_structure.layers.back();
    cur_layer->param                 = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->name                      = cur_layer->name;
    param->type                      = cur_layer->type_str;
    param->quantized                 = quantized_model;
    param->reshape_type              = 1;
    param->axis                      = 0;
    // tensorflow reshape (n,h,w,c);
    std::vector<int> reshape_dim;

    const auto option = tf_lite_operator->builtin_options.AsReshapeOptions();
    if (option == nullptr || option->new_shape.empty()) {
        ASSERT(tf_lite_operator->inputs.size() == 2);
        const auto& new_shape_tensor = tf_lite_tensors[tf_lite_operator->inputs[1]];
        ASSERT(new_shape_tensor->type == tflite::TensorType_INT32);
        const auto& buffer = tf_lite_model_buffer[new_shape_tensor->buffer];
        int data_count     = buffer->data.size() / SizeofTFLiteTensorData(new_shape_tensor->type);
        if (data_count != 0) {
            reshape_dim.assign(reinterpret_cast<int32_t*>(buffer->data.data()),
                               reinterpret_cast<int32_t*>(buffer->data.data()) + data_count);
            reshape_dim[0] = 0;
            ConvertShapeFormatTFLite(reshape_dim);
            param->num_axes = reshape_dim.size();
            param->shape    = reshape_dim;
            cur_layer->inputs.resize(1);
            cur_layer->inputs[0] = tf_lite_tensors[tf_lite_operator->inputs[0]]->name;
        } else {
            LOGE("TFLite do not support this type reshape!\n");
            return TNN_NS::Status(TNN_NS::TNNERR_UNSUPPORT_NET, "TFLite do not support this type reshape!\n");
        }
    } else {
        const auto& new_shape = option->new_shape;
        reshape_dim.assign(new_shape.begin(), new_shape.end());
        reshape_dim[0] = 0;
        ConvertShapeFormatTFLite(reshape_dim);
        param->num_axes = reshape_dim.size();
        param->shape    = reshape_dim;
        cur_layer->inputs.resize(1);
        cur_layer->inputs[0] = tf_lite_tensors[tf_lite_operator->inputs[0]]->name;
    }
    if (quantized_model) {
        // handle input blob scale
        auto input_index = tf_lite_operator->inputs[0];
        auto status      = CreateBlobScaleResource(net_resource, tf_lite_tensors, input_index);
        ASSERT(status == TNN_NS::TNN_CONVERT_OK);
        // handle output blob scale
        auto output_index = tf_lite_operator->outputs[0];
        status            = CreateBlobScaleResource(net_resource, tf_lite_tensors, output_index);
        ASSERT(status == TNN_NS::TNN_CONVERT_OK);
    }
    return TNN_NS::TNN_CONVERT_OK;
}

using namespace tflite;
REGISTER_CONVERTER(Reshape, BuiltinOperator_RESHAPE);
}  // namespace TNN_CONVERTER
