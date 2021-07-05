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

#include "onnx_base_converter.h"
#include "onnx_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_CONVERTER {
DECLARE_OP_CONVERTER(Reshape);

std::string OnnxReshapeConverter::TNNOpType(const onnx::NodeProto &node, bool quantized_model) {
    return "Reshape";
}
TNN_NS::ActivationType OnnxReshapeConverter::ActivationType(const onnx::NodeProto &node) {
    return TNN_NS::ActivationType_None;
}
TNN_NS::Status OnnxReshapeConverter::exec(TNN_NS::NetStructure &net_structure, TNN_NS::NetResource &net_resource,
                                          const onnx::NodeProto &node,
                                          std::map<std::string, const onnx::TensorProto *> &proxy_initializers_map,
                                          std::map<std::string, std::shared_ptr<OnnxProxyNode>> &proxy_nodes,
                                          bool &quantized_model) {
    const std::string &onnx_op = node.op_type();
    auto param                 = new TNN_NS::ReshapeLayerParam;
    auto cur_layer             = net_structure.layers.back();
    cur_layer->param           = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->type                = cur_layer->type_str;
    param->name                = cur_layer->name;
    param->quantized           = false;
#if 0
    param->axis         = 0;
    param->num_axes     = 4;
    param->shape        = {0, -1, 1, 1};
    param->reshape_type = 0;
    return TNN_NS::TNN_CONVERT_OK;
#endif
    const auto &resource_map = net_resource.resource_map;
    const auto &shape_name   = node.input(1);
    if (proxy_initializers_map.find(shape_name) != proxy_initializers_map.end()) {
        const auto shape_tensor     = proxy_initializers_map[shape_name];
        std::vector<int> shape_dims = std::vector<int>(shape_tensor->dims().begin(), shape_tensor->dims().end());
        std::vector<int> shape;
        int shape_count    = TNN_NS::DimsVectorUtils::Count(shape_dims);
        void *raw_data_ptr = GetDataFromTensor(*shape_tensor, onnx::TensorProto_DataType_INT64);
        for (int i = 0; i < shape_count; ++i) {
            shape.push_back(*((int64_t *)raw_data_ptr + i));
        }
        param->axis         = 0;
        param->num_axes     = shape.size();
        param->shape        = shape;
        param->reshape_type = 0;
        cur_layer->inputs.resize(1);
        cur_layer->inputs[0] = node.input(0);
    } else {
        param->axis         = 0;
        param->num_axes     = 0;
        param->shape        = {};
        param->reshape_type = 0;
    }

    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_CONVERTER(Reshape, Reshape);
}  // namespace TNN_CONVERTER
