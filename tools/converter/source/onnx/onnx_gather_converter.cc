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

#include <memory>

#include "onnx_base_converter.h"
#include "onnx_utils.h"
#include "tnn/interpreter/layer_resource.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_CONVERTER {

DECLARE_OP_CONVERTER(Gather);

std::string OnnxGatherConverter::TNNOpType(const onnx::NodeProto &node, bool quantized_model) {
    return "Gather";
}

TNN_NS::ActivationType OnnxGatherConverter::ActivationType(const onnx::NodeProto &node) {
    return TNN_NS::ActivationType_None;
}

TNN_NS::Status OnnxGatherConverter::exec(TNN_NS::NetStructure &net_structure, TNN_NS::NetResource &net_resource,
                                         const onnx::NodeProto &node,
                                         std::map<std::string, const onnx::TensorProto *> &proxy_initializers_map,
                                         std::map<std::string, std::shared_ptr<OnnxProxyNode>> &proxy_nodes,
                                         bool &quantized_model) {
    ASSERT(node.input_size() == 2);
    auto param       = new TNN_NS::GatherLayerParam;
    auto cur_layer   = net_structure.layers.back();
    cur_layer->param = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->type      = cur_layer->type_str;
    param->name      = cur_layer->name;
    param->quantized = false;
    param->axis      = GetAttributeInt(node, "axis", 0);

    auto &resource_map            = net_resource.resource_map;
    auto resource                 = std::make_shared<TNN_NS::GatherLayerResource>();
    resource_map[cur_layer->name] = resource;
    // parse indices
    const auto &indices_name = node.input(1);
    if (proxy_initializers_map.find(indices_name) != proxy_initializers_map.end()) {
        param->indices_in_resource            = true;
        auto indices_tensor                   = proxy_initializers_map[indices_name];
        TNN_NS::RawBuffer *indices_raw_buffer = nullptr;
        CreateRawBufferFromTensor(*indices_tensor, &indices_raw_buffer);
        resource->indices = *indices_raw_buffer;
    } else if (proxy_nodes.find(indices_name) != proxy_nodes.end() &&
               proxy_nodes.find(indices_name)->second->op_type == "Constant") {
        param->indices_in_resource            = true;
        auto indices_node                     = proxy_nodes.find(indices_name)->second->onnx_node;
        TNN_NS::RawBuffer *indices_raw_buffer = nullptr;
        CreateRawBufferFromConstant(*indices_node, &indices_raw_buffer);
        ASSERT(indices_raw_buffer != nullptr);
        resource->indices = *indices_raw_buffer;
    } else {
        param->indices_in_resource = false;
    }
    // parse data
    const auto &data_name = node.input(0);
    if (proxy_initializers_map.find(data_name) != proxy_initializers_map.end()) {
        param->data_in_resource = true;
        auto data_tensor        = proxy_initializers_map[data_name];
        auto &data_dims         = data_tensor->dims();
        auto dims               = std::vector<int>(data_dims.begin(), data_dims.end());
        auto data_count         = TNN_NS::DimsVectorUtils::Count(dims);
        auto data_raw_buffer    = TNN_NS::RawBuffer(data_count * sizeof(float), dims);
        data_raw_buffer.SetDataType(TNN_NS::DATA_TYPE_FLOAT);
        void *data_ptr = GetDataFromTensor(*data_tensor, onnx::TensorProto_DataType_FLOAT);
        if (data_ptr == nullptr) {
            LOGE("Gather: can not get data from onnx model, please check the model\n");
            return TNN_NS::TNNERR_MODEL_ERR;
        }
        auto tmp = new float[data_count];
        for (int i = 0; i < data_count; ++i) {
            tmp[i] = *((float *)data_ptr + i);
        }
        memcpy(data_raw_buffer.force_to<float *>(), tmp, data_count * sizeof(float));
        resource->data = data_raw_buffer;
        delete[] tmp;
    } else {
        param->data_in_resource = false;
    }
    // check indices and
    if (param->indices_in_resource && param->indices_in_resource == param->data_in_resource) {
        LOGE("Gather: There is not possible indices and data both in  or both not in resource\n");
        return TNN_NS::TNNERR_MODEL_ERR;
    }
    if (!param->data_in_resource && param->indices_in_resource) {
        cur_layer->inputs.resize(1);
        cur_layer->inputs[0] = node.input(0);
    } else if (param->data_in_resource && !param->indices_in_resource) {
        cur_layer->inputs.resize(1);
        cur_layer->inputs[0] = node.input(1);
    }
    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_CONVERTER(Gather, Gather);

}  // namespace TNN_CONVERTER
