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
#include "tools/converter/source/onnx/onnx_base_converter.h"

namespace TNN_CONVERTER {
DECLARE_OP_CONVERTER(Split);

std::string OnnxSplitConverter::TNNOpType(const onnx::NodeProto &node, bool quantized_model) {
    return "SplitV";
}

TNN_NS::ActivationType OnnxSplitConverter::ActivationType(const onnx::NodeProto &node) {
    return TNN_NS::ActivationType_None;
}

TNN_NS::Status OnnxSplitConverter::exec(TNN_NS::NetStructure &net_structure, TNN_NS::NetResource &net_resource,
                                        const onnx::NodeProto &node,
                                        std::map<std::string, const onnx::TensorProto *> &proxy_initializers_map,
                                        std::map<std::string, std::shared_ptr<OnnxProxyNode>> &proxy_nodes,
                                        bool &quantized_model) {
    const std::string &onnx_op = node.op_type();
    auto param                 = new TNN_NS::SplitVLayerParam;
    auto cur_layer             = net_structure.layers.back();
    cur_layer->param           = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->type                = cur_layer->type_str;
    param->name                = cur_layer->name;
    param->quantized           = false;
    param->axis                = GetAttributeInt(node, "axis", 1);
    param->slices              = GetAttributeIntVector(node, "split");
    if (node.input().size() > 1) {
        std::string split_name = node.input(1);
        const auto split_tensor     = proxy_initializers_map[split_name];
        std::vector<int> dims = std::vector<int>(split_tensor->dims().begin(), split_tensor->dims().end());
        int shape_count    = TNN_NS::DimsVectorUtils::Count(dims);
        void *raw_data_ptr = GetDataFromTensor(*split_tensor, onnx::TensorProto_DataType_INT64);
        for (int i = 0; i < shape_count; ++i) {
            param->slices.push_back(*((int64_t *)raw_data_ptr + i));
        }
    }
    cur_layer->inputs.resize(1);
    cur_layer->inputs[0] = node.input(0);
    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_CONVERTER(Split, Split);

}  // namespace TNN_CONVERTER
