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
#include "tnn/interpreter/raw_buffer.h"

namespace TNN_CONVERTER {
DECLARE_OP_CONVERTER(Gemm);

std::string OnnxGemmConverter::TNNOpType(const onnx::NodeProto &node, bool quantized_model) {
    float alpha = GetAttributeFloat(node, "alpha", 1.f);
    int transA  = (int)GetAttributeInt(node, "transA", 0);
    if (alpha == 1.f) {
        // InnerProduct-like A * B + C
        if (transA == 0) {
            return "InnerProduct";
        }
    }
    return "";
}
TNN_NS::ActivationType OnnxGemmConverter::ActivationType(const onnx::NodeProto &node) {
    return TNN_NS::ActivationType_None;
}
TNN_NS::Status OnnxGemmConverter::exec(TNN_NS::NetStructure &net_structure, TNN_NS::NetResource &net_resource,
                                       const onnx::NodeProto &node,
                                       std::map<std::string, const onnx::TensorProto *> &proxy_initializers_map,
                                       std::map<std::string, std::shared_ptr<OnnxProxyNode>> &proxy_nodes,
                                       bool &quantized_model) {
    const int input_size = node.input_size();
    assert(input_size == 2 || input_size == 3);
    auto *param    = new TNN_NS::InnerProductLayerParam;
    auto cur_layer = net_structure.layers.back();

    float alpha = GetAttributeFloat(node, "alpha", 1.f);
    float beta  = GetAttributeFloat(node, "beta", 1.f);
    int transA  = (int)GetAttributeInt(node, "transA", 0);
    int transB  = (int)GetAttributeInt(node, "transB", 0);

    assert(beta == 1 || beta == 0);
    assert(alpha == 1.f && transA == 0);

    // get gemm param
    const auto &weight_name   = node.input(1);
    const auto *weight_tensor = proxy_initializers_map[weight_name];
    float *weight_tensor_data = (float *)GetTensorProtoData(*weight_tensor);
    int weight_tensor_size    = GetTensorProtoDataSize(*weight_tensor);
    auto weight_dims          = CreateDimsVectorFromTensor(*weight_tensor);

    const auto h = weight_tensor->dims(0);
    const auto w = weight_tensor->dims(1);
    if (transB != 1) {
        auto *permuted_data      = new float[h * w];
        float *permuted_data_ptr = permuted_data;
        for (int j = 0; j < w; j++) {
            for (int k = 0; k < h; k++) {
                float vb           = weight_tensor_data[k * w + j];
                *permuted_data_ptr = vb;
                permuted_data_ptr++;
            }
        }
        ::memcpy(weight_tensor_data, permuted_data, weight_tensor_size * sizeof(float));
        delete[] permuted_data;
    }

    cur_layer->param  = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->name       = cur_layer->name;
    param->type       = cur_layer->type_str;
    param->quantized  = false;
    param->axis       = 1;
    param->transpose  = 0;
    param->num_output = weight_tensor->dims(0);

    auto layer_resource             = new TNN_NS::InnerProductLayerResource;
    layer_resource->name            = cur_layer->name;
    TNN_NS::RawBuffer weight_handle = TNN_NS::RawBuffer(weight_tensor_size * sizeof(float));
    weight_handle.SetDataType(TNN_NS::DATA_TYPE_FLOAT);
    weight_handle.SetBufferDims(weight_dims);
    ::memcpy(weight_handle.force_to<float *>(), weight_tensor_data, weight_tensor_size * sizeof(float));
    layer_resource->weight_handle = weight_handle;

    if (input_size > 2) {
        // Get Bias
        param->has_bias       = 1;
        const auto &bias_name = node.input(2);

        const auto *bias_tensor = proxy_initializers_map[bias_name];
        auto *bias_tensor_data  = (float *)GetTensorProtoData(*bias_tensor);
        int bias_tensor_size    = GetTensorProtoDataSize(*bias_tensor);
        auto bias_dims          = CreateDimsVectorFromTensor(*bias_tensor);

        TNN_NS::RawBuffer bias_handle = TNN_NS::RawBuffer(bias_tensor_size * sizeof(float));
        bias_handle.SetDataType(TNN_NS::DATA_TYPE_FLOAT);
        bias_handle.SetBufferDims(bias_dims);
        ::memcpy(bias_handle.force_to<float *>(), bias_tensor_data, bias_tensor_size * sizeof(float));
        layer_resource->bias_handle = bias_handle;

        for (int i = 0; i < bias_tensor_size; i++) {
            float tmp = layer_resource->bias_handle.force_to<float *>()[i];
            int x     = 0;
        }
    }
    net_resource.resource_map[cur_layer->name] = std::shared_ptr<TNN_NS::LayerResource>(layer_resource);
    int size                                   = net_resource.resource_map.size();

    cur_layer->inputs.resize(1);
    cur_layer->inputs[0] = node.input(0);
    cur_layer->outputs.resize(1);
    cur_layer->outputs[0] = node.output(0);

    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_CONVERTER(Gemm, Gemm);
}  // namespace TNN_CONVERTER
