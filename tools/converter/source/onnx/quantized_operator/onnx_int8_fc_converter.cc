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

#include "onnx/onnx_base_converter.h"
#include "onnx/onnx_utils.h"
#include "tnn/interpreter/raw_buffer.h"
#include "tnn/interpreter/tnn/objseri.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_CONVERTER {

DECLARE_OP_CONVERTER(Int8InnerProduct);

std::string OnnxInt8InnerProductConverter::TNNOpType(const onnx::NodeProto &node, bool quantized_model) {
    return "QuantizedInnerProduct";
}

TNN_NS::ActivationType OnnxInt8InnerProductConverter::ActivationType(const onnx::NodeProto &node) {
    return TNN_NS::ActivationType_None;
}

TNN_NS::Status OnnxInt8InnerProductConverter::exec(
    TNN_NS::NetStructure &net_structure, TNN_NS::NetResource &net_resource, const onnx::NodeProto &node,
    std::map<std::string, const onnx::TensorProto *> &proxy_initializers_map,
    std::map<std::string, std::shared_ptr<OnnxProxyNode>> &proxy_nodes, bool &quantized_model) {
    const int input_size = node.input_size();
    assert(input_size == 2 || input_size == 3);
    auto *param      = new TNN_NS::InnerProductLayerParam;
    auto cur_layer   = net_structure.layers.back();
    cur_layer->param = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->name      = cur_layer->name;
    param->type      = cur_layer->type_str;
    param->quantized = true;
    param->axis      = 1;
    param->transpose = 0;
    // get convolution param
    const auto &weight_name = node.input(1);
    const auto &weight_node = FindNodeProto(weight_name, proxy_nodes);
    auto weight_shape       = GetAttributeIntVector(*weight_node, "shape");
    assert(weight_shape.size() == 2);
    auto co           = weight_shape[0];
    param->num_output = co;

    // create input blob scale
    const auto &input_name     = node.input(0);
    const auto &input_node     = FindNodeProto(input_name, proxy_nodes);
    auto input_scale           = GetAttributeFloat(*input_node, "Y_scale", 1.0f);
    auto input_zero_point      = GetAttributeInt(*input_node, "Y_zero_point", 0);
    auto input_blob_scale_name = input_name + BLOB_SCALE_SUFFIX;
    if (net_resource.resource_map.find(input_blob_scale_name) == net_resource.resource_map.end()) {
        // create input blob scale
        // assert(input_zero_point == 0);
        auto input_blob_scale                = new TNN_NS::IntScaleResource;
        input_blob_scale->name               = input_blob_scale_name;
        TNN_NS::RawBuffer input_scale_handle = TNN_NS::RawBuffer(1 * sizeof(float), (char *)&input_scale);
        input_scale_handle.SetDataType(TNN_NS::DATA_TYPE_FLOAT);
        input_scale_handle.SetBufferDims({1});
        input_blob_scale->scale_handle = input_scale_handle;
        TNN_NS::RawBuffer bias_handle  = TNN_NS::RawBuffer(1 * sizeof(int32_t), (char *)&input_zero_point);
        bias_handle.SetDataType(TNN_NS::DATA_TYPE_INT32);
        bias_handle.SetBufferDims({1});
        input_blob_scale->bias_handle       = bias_handle;
        TNN_NS::RawBuffer zero_point_handle = TNN_NS::RawBuffer(1 * sizeof(int8_t));
        zero_point_handle.SetDataType(TNN_NS::DATA_TYPE_INT8);
        zero_point_handle.SetBufferDims({1});
        input_blob_scale->zero_point_handle              = zero_point_handle;
        net_resource.resource_map[input_blob_scale_name] = std::shared_ptr<TNN_NS::LayerResource>(input_blob_scale);
    }

    // quantized weight value
    auto weight_scale            = GetAttributeFloat(*weight_node, "Y_scale", 1.0);
    auto weight_zero_point       = GetAttributeInt(*weight_node, "Y_zero_point", 0);
    auto asymmetric_weight_value = GetAttributeUInt8Vector(*weight_node, "values");
    auto weight_value            = Asymmetric2Symmetric(asymmetric_weight_value, weight_zero_point);
    auto weight_count            = weight_shape[0] * weight_shape[1];
    assert(weight_count == weight_value.size());
    auto layer_resource             = new TNN_NS::InnerProductLayerResource;
    layer_resource->name            = cur_layer->name;
    TNN_NS::RawBuffer weight_handle = TNN_NS::RawBuffer(weight_count * sizeof(uint8_t));
    weight_handle.SetDataType(TNN_NS::DATA_TYPE_INT8);
    weight_handle.SetBufferDims({weight_shape[0], weight_shape[1]});
    ::memcpy(weight_handle.force_to<uint8_t *>(), weight_value.data(), weight_count * sizeof(uint8_t));
    layer_resource->weight_handle = weight_handle;
    // quantized weight scale
    auto cal_weight_scale          = input_scale * weight_scale;
    TNN_NS::RawBuffer scale_handle = TNN_NS::RawBuffer(1 * sizeof(float), (char *)&cal_weight_scale);
    scale_handle.SetDataType(TNN_NS::DATA_TYPE_FLOAT);
    scale_handle.SetBufferDims({1});
    layer_resource->scale_handle        = scale_handle;
    TNN_NS::RawBuffer zero_point_handle = TNN_NS::RawBuffer(1 * sizeof(int8_t));
    zero_point_handle.SetDataType(TNN_NS::DATA_TYPE_INT8);
    zero_point_handle.SetBufferDims({1});
    layer_resource->zero_point_handle = zero_point_handle;

    if (input_size > 2) {
        // Get Bias
        param->has_bias       = 1;
        const auto &bias_name = node.input(2);
        const auto &bias_node = FindNodeProto(bias_name, proxy_nodes);
        auto bias_scale       = GetAttributeFloat(*bias_node, "Y_scale", 1.0);
        auto bias_zero_point  = GetAttributeInt(*bias_node, "Y_zero_point", 0);
        auto bias_shape       = GetAttributeIntVector(*bias_node, "shape");
        auto bias_value       = GetAttributeIntVector(*bias_node, "values");
        // calculate bias
        std::vector<int32_t> cal_bias_value;
        for (const auto value : bias_value) {
            cal_bias_value.push_back(value * bias_scale / cal_weight_scale);
        }
        assert(bias_shape.size() == 1);
        assert(bias_zero_point == 0);
        TNN_NS::RawBuffer bias_handle = TNN_NS::RawBuffer(cal_bias_value.size() * sizeof(int32_t));
        bias_handle.SetDataType(TNN_NS::DATA_TYPE_INT32);
        TNN_NS::DimsVector bias_dims;
        bias_dims.push_back(cal_bias_value.size());
        bias_handle.SetBufferDims(bias_dims);
        ::memcpy(bias_handle.force_to<int32_t *>(), cal_bias_value.data(), bias_value.size() * sizeof(int32_t));
        layer_resource->bias_handle = bias_handle;
    }
    // update net_resource resource_map
    net_resource.resource_map[cur_layer->name] = std::shared_ptr<TNN_NS::LayerResource>(layer_resource);

    // create output blob_scale
    const auto &output_name    = node.output(0);
    auto output_blob_cale_name = output_name + BLOB_SCALE_SUFFIX;
    if (net_resource.resource_map.find(output_blob_cale_name) == net_resource.resource_map.end()) {
        auto output_scale                     = GetAttributeFloat(node, "Y_scale", 1.0);
        auto output_zero_point                = GetAttributeInt(node, "Y_zero_point", 0);
        auto output_blob_scale                = new TNN_NS::IntScaleResource;
        output_blob_scale->name               = output_blob_cale_name;
        TNN_NS::RawBuffer output_scale_handle = TNN_NS::RawBuffer(1 * sizeof(float), (char *)&output_scale);
        output_scale_handle.SetDataType(TNN_NS::DATA_TYPE_FLOAT);
        output_scale_handle.SetBufferDims({1});
        output_blob_scale->scale_handle = output_scale_handle;
        TNN_NS::RawBuffer bias_handle   = TNN_NS::RawBuffer(1 * sizeof(int32_t), (char *)&output_zero_point);
        bias_handle.SetDataType(TNN_NS::DATA_TYPE_INT32);
        bias_handle.SetBufferDims({1});
        output_blob_scale->bias_handle      = bias_handle;
        TNN_NS::RawBuffer zero_point_handle = TNN_NS::RawBuffer(1 * sizeof(int8_t));
        zero_point_handle.SetDataType(TNN_NS::DATA_TYPE_INT8);
        zero_point_handle.SetBufferDims({1});
        output_blob_scale->zero_point_handle             = zero_point_handle;
        net_resource.resource_map[output_blob_cale_name] = std::shared_ptr<TNN_NS::LayerResource>(output_blob_scale);
    }
    cur_layer->inputs.resize(1);
    cur_layer->inputs[0] = node.input(0);
    cur_layer->outputs.resize(1);
    cur_layer->outputs[0] = node.output(0);
    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_CONVERTER(Int8InnerProduct, Int8FC);

}  // namespace TNN_CONVERTER
