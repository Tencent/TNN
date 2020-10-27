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

DECLARE_OP_CONVERTER(Resize);

std::string TFLiteResizeConverter::TNNOpType(tflite::BuiltinOperator op_code, bool quantized_model) {
    return "Upsample";
}

tflite::ActivationFunctionType TFLiteResizeConverter::ActivationType(
    const std::unique_ptr<tflite::OperatorT> &tf_lite_operator, tflite::BuiltinOperator op_code) {
    return tflite::ActivationFunctionType_NONE;
}

TNN_NS::Status TFLiteResizeConverter::exec(TNN_NS::NetStructure &net_structure, TNN_NS::NetResource &net_resource,
                                           const std::unique_ptr<tflite::OperatorT> &tf_lite_operator,
                                           const std::vector<std::unique_ptr<tflite::TensorT>> &tf_lite_tensors,
                                           const std::vector<std::unique_ptr<tflite::BufferT>> &tf_lite_model_buffer,
                                           const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tf_lite_op_set,
                                           bool quantized_model) {
    auto param       = new TNN_NS::UpsampleLayerParam;
    auto cur_layer   = net_structure.layers.back();
    cur_layer->param = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->name      = cur_layer->name;
    param->type      = cur_layer->type_str;
    param->quantized = false;
    // scales always 1.0
    param->scales        = {1.0, 1.0};
    auto tf_lite_op_type = tf_lite_op_set[tf_lite_operator->opcode_index]->builtin_code;
    if (tf_lite_op_type == tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR) {
        param->mode          = 1;
        auto option          = tf_lite_operator->builtin_options.AsResizeNearestNeighborOptions();
        param->align_corners = option->align_corners;
    } else if (tf_lite_op_type == tflite::BuiltinOperator_RESIZE_BILINEAR) {
        param->mode          = 2;
        auto option          = tf_lite_operator->builtin_options.AsResizeBilinearOptions();
        param->align_corners = option->align_corners;
    } else {
        LOGE("TNN TFLite Converter don't support resize mode!\n");
        return TNN_NS::TNNERR_CONVERT_UNSUPPORT_LAYER;
    }
    ASSERT(tf_lite_operator->inputs.size() == 2);
    auto &output_shape_tensor = tf_lite_tensors[tf_lite_operator->inputs[1]];
    auto output_shape_data_ptr =
        reinterpret_cast<int *>(tf_lite_model_buffer[output_shape_tensor->buffer]->data.data());
    auto data_size = SizeofTFLiteTensorData(output_shape_tensor->type);
    ASSERT(tf_lite_model_buffer[output_shape_tensor->buffer]->data.size() / data_size == 2);
    int output_shape_height = output_shape_data_ptr[0];
    int output_shape_width  = output_shape_data_ptr[1];
    param->dims.push_back(output_shape_width);
    param->dims.push_back(output_shape_height);

    cur_layer->inputs.resize(1);
    cur_layer->inputs[0] = tf_lite_tensors[tf_lite_operator->inputs[0]]->name;
    return TNN_NS::TNN_CONVERT_OK;
}

using namespace tflite;
REGISTER_CONVERTER(Resize, BuiltinOperator_RESIZE_BILINEAR);
REGISTER_CONVERTER(Resize, BuiltinOperator_RESIZE_NEAREST_NEIGHBOR);
}  // namespace TNN_CONVERTER
