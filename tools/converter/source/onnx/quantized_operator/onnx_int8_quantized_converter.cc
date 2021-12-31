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

#include "onnx/onnx_utils.h"
#include "tnn/interpreter/raw_buffer.h"
#include "tnn/interpreter/tnn/objseri.h"
#include "tools/converter/source/onnx/onnx_base_converter.h"

namespace TNN_CONVERTER {
DECLARE_OP_CONVERTER(Int8Quantized);

std::string OnnxInt8QuantizedConverter::TNNOpType(const onnx::NodeProto &node, bool quantized_model) {
    return "Int8Quantized";
}

TNN_NS::ActivationType OnnxInt8QuantizedConverter::ActivationType(const onnx::NodeProto &node) {
    return TNN_NS::ActivationType_None;
}

TNN_NS::Status OnnxInt8QuantizedConverter::exec(
    TNN_NS::NetStructure &net_structure, TNN_NS::NetResource &net_resource, const onnx::NodeProto &node,
    std::map<std::string, const onnx::TensorProto *> &proxy_initializers_map,
    std::map<std::string, std::shared_ptr<OnnxProxyNode>> &proxy_nodes, bool &quantized_model) {
    TNN_NS::LayerParam *param = new TNN_NS::LayerParam;
    auto cur_layer            = net_structure.layers.back();
    cur_layer->param          = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->name               = cur_layer->name;
    param->type               = cur_layer->type_str;
    param->quantized          = false;
    for (int i = 0; i < node.input_size(); ++i) {
        const auto &input_name            = node.input(i);
        std::string input_blob_scale_name = input_name + BLOB_SCALE_SUFFIX;
        auto &resource_map                = net_resource.resource_map;
        if (resource_map.find(input_blob_scale_name) != resource_map.end()) {
            continue;
        }
        auto scale                           = GetAttributeFloat(node, "Y_scale", 1.0);
        auto zero_point                      = GetAttributeInt(node, "Y_zero_point", 0);
        auto input_blob_scale                = new TNN_NS::IntScaleResource;
        input_blob_scale->name               = input_blob_scale_name;
        TNN_NS::RawBuffer input_scale_handle = TNN_NS::RawBuffer(1 * sizeof(float), (char *)&scale);
        input_scale_handle.SetDataType(TNN_NS::DATA_TYPE_FLOAT);
        input_blob_scale->scale_handle = input_scale_handle;
        TNN_NS::RawBuffer bias_handle  = TNN_NS::RawBuffer(1 * sizeof(int32_t), (char *)&zero_point);
        bias_handle.SetDataType(TNN_NS::DATA_TYPE_INT32);
        input_blob_scale->bias_handle = bias_handle;
        TNN_NS::RawBuffer zero_point_handle = TNN_NS::RawBuffer(1 * sizeof(int8_t));
        zero_point_handle.SetDataType(TNN_NS::DATA_TYPE_INT8);
        input_blob_scale->zero_point_handle              = zero_point_handle;
        net_resource.resource_map[input_blob_scale_name] = std::shared_ptr<TNN_NS::LayerResource>(input_blob_scale);
    }

    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_CONVERTER(Int8Quantized, Int8Quantize);

}  // namespace TNN_CONVERTER
