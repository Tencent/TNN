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
#include "tnn/interpreter/tnn/objseri.h"
#include "tools/converter/source/onnx/onnx_base_converter.h"

namespace TNN_CONVERTER {
DECLARE_OP_CONVERTER(Int8Dequantized);

std::string OnnxInt8DequantizedConverter::TNNOpType(const onnx::NodeProto &node, bool quantized_model) {
    return "Int8Dequantized";
}

TNN_NS::ActivationType OnnxInt8DequantizedConverter::ActivationType(const onnx::NodeProto &node) {
    return TNN_NS::ActivationType_None;
}

TNN_NS::Status OnnxInt8DequantizedConverter::exec(
    tnn::NetStructure &net_structure, tnn::NetResource &net_resource, const onnx::NodeProto &node,
    std::map<std::string, const onnx::TensorProto *> proxy_initializers_map,
    std::map<std::string, std::shared_ptr<OnnxProxyNode>> proxy_nodes, bool &quantized_model) {
#if 0
    const auto &output_name     = node.output(0);
    auto output_blob_scale_name = output_name + BLOB_SCALE_SUFFIX;
    if (net_resource.resource_map.find(output_blob_scale_name) == net_resource.resource_map.end()) {
        auto scale                     = GetAttributeFloat(node, "Y_scale", 1.0);
        auto zero_point                = GetAttributeInt(node, "Y_zero_point", 0);
        auto output_blob_scale         = new TNN_NS::IntScaleResource;
        output_blob_scale->name        = output_blob_scale_name;
        TNN_NS::RawBuffer scale_handle = TNN_NS::RawBuffer(1 * sizeof(float), (char *)&scale);
        scale_handle.SetDataType(TNN_NS::DATA_TYPE_FLOAT);
        output_blob_scale->scale_handle     = scale_handle;
        TNN_NS::RawBuffer zero_point_handle = TNN_NS::RawBuffer(1 * sizeof(int32_t), (char *)&zero_point);
        zero_point_handle.SetDataType(TNN_NS::DATA_TYPE_INT32);
        output_blob_scale->bias_handle                    = zero_point_handle;
        net_resource.resource_map[output_blob_scale_name] = std::shared_ptr<TNN_NS::LayerResource>(output_blob_scale);
    }
#endif
    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_CONVERTER(Int8Dequantized, Int8Dequantize);

}  // namespace TNN_CONVERTER