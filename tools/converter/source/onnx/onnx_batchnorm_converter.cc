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
DECLARE_OP_CONVERTER(BatchNorm);

std::string OnnxBatchNormConverter::TNNOpType(const onnx::NodeProto &node, bool quantized_model) {
    return "BatchNormCxx";
}

TNN_NS::ActivationType OnnxBatchNormConverter::ActivationType(const onnx::NodeProto &node) {
    return TNN_NS::ActivationType_None;
}

TNN_NS::Status OnnxBatchNormConverter::exec(TNN_NS::NetStructure &net_structure, TNN_NS::NetResource &net_resource,
                                            const onnx::NodeProto &node,
                                            std::map<std::string, const onnx::TensorProto *> &proxy_initializers_map,
                                            std::map<std::string, std::shared_ptr<OnnxProxyNode>> &proxy_nodes,
                                            bool &quantized_model) {
    const std::string &onnx_op = node.op_type();
    auto param                 = new TNN_NS::BatchNormLayerParam;
    auto cur_layer             = net_structure.layers.back();
    cur_layer->param           = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->type                = cur_layer->type_str;
    param->name                = cur_layer->name;
    param->quantized           = false;

    const double epsilon = GetAttributeFloat(node, "epsilon", 1e-5f);

    const auto &gamma = proxy_initializers_map[node.input(1)];
    const auto &beta  = proxy_initializers_map[node.input(2)];
    const auto &mean  = proxy_initializers_map[node.input(3)];
    const auto &var   = proxy_initializers_map[node.input(4)];

    const auto *gamma_data = GetTensorProtoData(*gamma);
    const auto *beta_data  = GetTensorProtoData(*beta);
    const auto *mean_data  = GetTensorProtoData(*mean);
    const auto *var_data   = GetTensorProtoData(*var);

    const int channels = GetTensorProtoDataSize(*gamma);
    auto *slope        = new float[channels];
    auto *bias         = new float[channels];

    double sqrt_var;
    for (int i = 0; i < channels; i++) {
        sqrt_var = std::sqrt(static_cast<double>(var_data[i]) + epsilon);
        slope[i] = static_cast<double>(gamma_data[i]) / sqrt_var;
        bias[i]  = static_cast<double>(beta_data[i]) - static_cast<double>(mean_data[i]) * slope[i];
    }

    auto layer_resource            = new TNN_NS::BatchNormLayerResource;
    layer_resource->name           = cur_layer->name;
    TNN_NS::RawBuffer scale_handle = TNN_NS::RawBuffer(channels * sizeof(float));
    scale_handle.SetDataType(TNN_NS::DATA_TYPE_FLOAT);
    scale_handle.SetBufferDims({channels});
    ::memcpy(scale_handle.force_to<float *>(), slope, channels * sizeof(float));
    layer_resource->scale_handle = scale_handle;

    TNN_NS::RawBuffer bias_handle = TNN_NS::RawBuffer(channels * sizeof(float));
    bias_handle.SetDataType(TNN_NS::DATA_TYPE_FLOAT);
    bias_handle.SetBufferDims({channels});
    ::memcpy(bias_handle.force_to<float *>(), bias, channels * sizeof(float));
    layer_resource->bias_handle = bias_handle;

    net_resource.resource_map[cur_layer->name] = std::shared_ptr<TNN_NS::LayerResource>(layer_resource);

    cur_layer->inputs.resize(1);
    cur_layer->inputs[0] = node.input(0);
    cur_layer->outputs.resize(1);
    cur_layer->outputs[0] = node.output(0);

    delete[] slope;
    delete[] bias;

    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_CONVERTER(BatchNorm, BatchNormalization);

}  // namespace TNN_CONVERTER
