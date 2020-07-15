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
DECLARE_OP_COVERTER(PRelu);

std::string TFLitePReluConverter::TNNOpType(bool quantizedModel) {
    return "PRelu";
}
TNN_NS::Status TFLitePReluConverter::exec(TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource,
                                          const std::unique_ptr<tflite::OperatorT>& tf_lite_operator,
                                          const std::vector<std::unique_ptr<tflite::TensorT>>& tf_lite_tensors,
                                          const std::vector<std::unique_ptr<tflite::BufferT>>& tf_lite_model_buffer,
                                          const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tf_lite_op_set,
                                          bool quantizedModel) {
    auto param =

        new TNN_NS::PReluLayerParam;
    auto cur_layer = net_structure.layers.back();

    // inputs: input tensor, weight
    const int input_size = tf_lite_operator->inputs.size();
    ASSERT(input_size == 2);
    // weight index
    const int weight_index    = tf_lite_operator->inputs[1];
    const auto& weight_tensor = tf_lite_tensors[weight_index];

    const auto& weight_shape = weight_tensor->shape;
    const int co             = weight_shape[2];

    param->name = cur_layer->name;
    param->type = cur_layer->type_str;

    // update param
    cur_layer->param = std::shared_ptr<TNN_NS::LayerParam>(param);

    // weight
    auto layer_resource = new TNN_NS::ConvLayerResource;
    TNN_NS::RawBuffer alpha_handle = TNN_NS::RawBuffer(co * sizeof(float));
    auto data_ptr = reinterpret_cast<const float*>(tf_lite_model_buffer[weight_tensor->buffer]->data.data());
    ::memcpy(alpha_handle.force_to<float*>(), data_ptr, sizeof(float) * co);

    return TNN_NS::TNN_CONVERT_OK;
}
using namespace tflite;
REGISTER_CONVERTER(PRelu, BuiltinOperator_PRELU);
}  // namespace TNN_CONVERTER
