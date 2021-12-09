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

DECLARE_OP_CONVERTER(Int8ConvRelu);

std::string OnnxInt8ConvReluConverter::TNNOpType(const onnx::NodeProto &node, bool quantized_model) {
    return "QuantizedConvolution";
}

TNN_NS::ActivationType OnnxInt8ConvReluConverter::ActivationType(const onnx::NodeProto &node) {
    return TNN_NS::ActivationType_None;
}

TNN_NS::Status OnnxInt8ConvReluConverter::exec(TNN_NS::NetStructure &net_structure, TNN_NS::NetResource &net_resource,
                                               const onnx::NodeProto &node,
                                               std::map<std::string, const onnx::TensorProto *> &proxy_initializers_map,
                                               std::map<std::string, std::shared_ptr<OnnxProxyNode>> &proxy_nodes,
                                               bool &quantized_model) {
    TNN_NS::ConvLayerParam *param = new TNN_NS::ConvLayerParam;
    auto cur_layer                = net_structure.layers.back();
    cur_layer->param              = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->name                   = cur_layer->name;
    param->type                   = cur_layer->type_str;
    param->quantized              = true;
    const int input_size          = node.input_size();
    ASSERT(input_size == 2 || input_size == 3);
    // get convolution param
    const auto &weight_name = node.input(1);
    const auto &weight_node = FindNodeProto(weight_name, proxy_nodes);
    auto weight_shape       = GetAttributeIntVector(*weight_node, "shape");
    auto group              = GetAttributeInt(node, "group", 1);
    const int co            = weight_shape[0];
    const int kh            = weight_shape[1];
    const int kw            = weight_shape[2];
    const int ci            = weight_shape[3];
    const int weight_count  = co * kw * kw * ci;
    param->input_channel    = ci;
    param->output_channel   = co;
    param->kernels.push_back(kw);
    param->kernels.push_back(kh);
    // onnx order: stride_h, stride_w
    // tnn  order: stride_w, stride_h
    auto strides = GetAttributeIntVector(node, "strides");
    ASSERT(strides.size() == 2);
    param->strides = {strides[1], strides[0]};
    // dilation
    auto dilations = GetAttributeIntVector(node, "dilations");
    ASSERT(dilations.size() == 2);
    param->dialations = {dilations[1], dilations[0]};
    param->group      = group;
    // parse pads type
    auto pads = GetAttributeIntVector(node, "pads");
    if (!pads.empty()) {
        param->pad_type = -1;
        if (pads[0] < pads[2] || pads[1] < pads[3]) {
            // same upper
            param->pad_type = 0;
        }
        param->pads = {pads[1], pads[3], pads[0], pads[2]};
    } else {
        auto auto_pad = GetAttributeString(node, "auto_pad", "NOTSET");
        if (auto_pad == "NOTSET") {
            param->pad_type = -1;
        } else if (auto_pad == "SAME_UPPER") {
            param->pad_type = 0;
        } else if (auto_pad == "VALID") {
            param->pad_type = 1;
        } else {
            LOGE("Conv: SAME_LOWER does not support, change toSAME_UPPER\n");
            return TNN_NS::TNNERR_CONVERT_UNSUPPORT_LAYER;
        }
        param->pads = {0, 0, 0, 0};
    }

    ASSERT(pads.size() == 4);
    if (node.op_type() == "Int8ConvRelu") {
        param->activation_type = TNN_NS::ActivationType_ReLU;
    } else if (node.op_type() == "Int8ConvRelu6") {
        param->activation_type = TNN_NS::ActivationType_ReLU6;
    } else {
        param->activation_type = TNN_NS::ActivationType_None;
    }

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
        input_blob_scale->scale_handle = input_scale_handle;
        TNN_NS::RawBuffer bias_handle  = TNN_NS::RawBuffer(1 * sizeof(int32_t), (char *)&input_zero_point);
        bias_handle.SetDataType(TNN_NS::DATA_TYPE_INT32);
        input_blob_scale->bias_handle = bias_handle;
        TNN_NS::RawBuffer zero_point_handle = TNN_NS::RawBuffer(1 * sizeof(int8_t));
        zero_point_handle.SetDataType(TNN_NS::DATA_TYPE_INT8);
        input_blob_scale->zero_point_handle = zero_point_handle;
        net_resource.resource_map[input_blob_scale_name] = std::shared_ptr<TNN_NS::LayerResource>(input_blob_scale);
    }

    // quantized weight value
    auto weight_scale      = GetAttributeFloat(*weight_node, "Y_scale", 1.0);
    auto weight_zero_point = GetAttributeInt(*weight_node, "Y_zero_point", 0);
    assert(weight_shape.size() == 4);
    auto asymmetric_weight_value = GetAttributeUInt8Vector(*weight_node, "values");
    auto weight_value            = Asymmetric2Symmetric(asymmetric_weight_value, weight_zero_point);
    assert(weight_value.size() == weight_count);
    auto layer_resource             = new TNN_NS::ConvLayerResource;
    layer_resource->name            = cur_layer->name;
    TNN_NS::RawBuffer filter_handle = TNN_NS::RawBuffer(weight_count * sizeof(int8_t));
    filter_handle.SetDataType(TNN_NS::DATA_TYPE_INT8);
    OHWI2OIHW(reinterpret_cast<int8_t *>(weight_value.data()), filter_handle.force_to<int8_t *>(), co, kh, kw, ci);
    layer_resource->filter_handle = filter_handle;
    // quantized weight scale
    auto cal_weight_scale          = input_scale * weight_scale;
    TNN_NS::RawBuffer scale_handle = TNN_NS::RawBuffer(1 * sizeof(float), (char *)&cal_weight_scale);
    scale_handle.SetDataType(TNN_NS::DATA_TYPE_FLOAT);
    layer_resource->scale_handle        = scale_handle;
    TNN_NS::RawBuffer zero_point_handle = TNN_NS::RawBuffer(1 * sizeof(int8_t));
    zero_point_handle.SetDataType(TNN_NS::DATA_TYPE_INT8);
    layer_resource->zero_point_handle = zero_point_handle;

    if (input_size > 2) {
        // Get Bias
        param->bias           = 1;
        const auto &bias_name = node.input(2);
        const auto &bias_node = FindNodeProto(bias_name, proxy_nodes);
        auto bias_scale       = GetAttributeFloat(*bias_node, "Y_scale", 1.0);
        auto bias_zero_point  = GetAttributeInt(*bias_node, "Y_zero_point", 0);
        auto bias_shape       = GetAttributeIntVector(*bias_node, "shape");
        auto bias_value       = GetAttributeIntVector(*bias_node, "values");
        // calculate bias
        std::vector<int32_t> cal_bias_value;
        for (const auto &value : bias_value) {
            cal_bias_value.push_back(value * bias_scale / cal_weight_scale);
        }
        assert(bias_shape.size() == 1);
        assert(bias_zero_point == 0);
        TNN_NS::RawBuffer bias_handle = TNN_NS::RawBuffer(cal_bias_value.size() * sizeof(int32_t));
        bias_handle.SetDataType(TNN_NS::DATA_TYPE_INT32);
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
        output_blob_scale->scale_handle = output_scale_handle;
        TNN_NS::RawBuffer bias_handle   = TNN_NS::RawBuffer(1 * sizeof(int32_t), (char *)&output_zero_point);
        bias_handle.SetDataType(TNN_NS::DATA_TYPE_INT32);
        output_blob_scale->bias_handle      = bias_handle;
        TNN_NS::RawBuffer zero_point_handle = TNN_NS::RawBuffer(1 * sizeof(int8_t));
        zero_point_handle.SetDataType(TNN_NS::DATA_TYPE_INT8);
        output_blob_scale->zero_point_handle             = zero_point_handle;
        net_resource.resource_map[output_blob_cale_name] = std::shared_ptr<TNN_NS::LayerResource>(output_blob_scale);
    }
    cur_layer->inputs.resize(1);
    cur_layer->inputs[0] = node.input(0);
    cur_layer->outputs.resize(1);
    cur_layer->outputs[0] = node.output(0);
    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_CONVERTER(Int8ConvRelu, Int8Conv);
REGISTER_CONVERTER(Int8ConvRelu, Int8ConvRelu);

}  // namespace TNN_CONVERTER
