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

#include "onnx_converter.h"

#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "onnx_base_converter.h"
#include "onnx_proxy_graph.h"
#include "onnx_utils.h"
#include "tnn/core/macro.h"

namespace TNN_CONVERTER {

Onnx2Tnn::Onnx2Tnn(std::string model_path) {
    this->onnx_model_path_ = model_path;
    this->onnx_model_      = std::unique_ptr<onnx::ModelProto>(new onnx::ModelProto());
}

Onnx2Tnn::~Onnx2Tnn() {
    // do nothing
}

TNN_NS::Status Onnx2Tnn::Converter2Tnn(TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource) {
    if (!ReadModel()) {
        return TNN_NS::TNNERR_CONVERT_INVALID_MODEL;
    }
    const auto& onnx_graph = onnx_model_->graph();
    std::shared_ptr<OnnxProxyGraph> onnx_proxy_graph(new OnnxProxyGraph(&onnx_graph));
    auto& proxy_initializers      = onnx_proxy_graph->proxy_initializers_map_;
    auto& proxy_inputs            = onnx_proxy_graph->proxy_inputs_map_;
    auto& proxy_outputs           = onnx_proxy_graph->proxy_outputs_map_;
    auto& proxy_nodes             = onnx_proxy_graph->proxy_nodes_map_;
    auto& constant_node_to_delete = onnx_proxy_graph->constant_node_to_delete_;

    bool quantized_mode = IsQuantized();
    // convert onnx graph input
    TNN_NS::InputShapesMap& input_shapes_map      = net_structure.inputs_shape_map;
    TNN_NS::InputDataTypeMap& input_data_type_map = net_structure.input_data_type_map;
    for (const auto iter : proxy_inputs) {
        // input in initializers
        if (proxy_initializers.find(iter.first) != proxy_initializers.end()) {
            continue;
        }
        auto input_name                = iter.first;
        auto input                     = iter.second;
        auto input_shape_tensor        = input->type().tensor_type().shape();
        TNN_NS::DimsVector dims_vector = ConvertTensorShapeProtoToDimsVector(input_shape_tensor);
        if (dims_vector.size() != 4) {
            // dims_vector.push_back(1);
            //            LOGE("The onnx have support input shape\n");
            //            return TNN_NS::TNNERR_CONVERT_INVALID_MODEL;
            continue;
        }
        if (input_shapes_map.find(input_name) == input_shapes_map.end()) {
            input_shapes_map[input_name] = dims_vector;
            const auto& input_data_type =
                static_cast<onnx::TensorProto_DataType>(input->type().tensor_type().elem_type());
            input_data_type_map[input_name] = GetTnnDataTypeFromOnnx(input_data_type);
        }
    }
    // convert onnx graph output
    std::set<std::string>& outputs = net_structure.outputs;
    for (const auto iter : proxy_outputs) {
        auto output_name = iter.first;
        if (outputs.find(output_name) == outputs.end()) {
            outputs.insert(output_name);
        }
    }

    // convert onnx nodes
    const auto node_size = onnx_graph.node_size();
    for (int i = 0; i < node_size; ++i) {
        const auto& node                = onnx_graph.node(i);
        const std::string& node_op_type = node.op_type();
        if (node_op_type == "Int8Quantize") {
            // TODO
            quantized_mode = true;
        } else if (node_op_type == "Int8Dequantize") {
            // TODO
            quantized_mode = false;
        } else if (node_op_type == "Int8GivenTensorFill" || node_op_type == "Int8GivenIntTensorFill") {
            continue;
        }

        if (node_op_type == "Constant") {
            continue;
        }

        auto converter = OnnxConverterManager::get()->search(node_op_type);
        if (converter == nullptr) {
            LOGE("The OnnxConverter do not support layer:%s output: %s \n", node.name().c_str(),
                 node.output(0).c_str());
            LOGE("The unsupported operator type is:%s\n", node_op_type.c_str());
            return TNN_NS::TNNERR_CONVERT_UNSUPPORT_LAYER;
        }
        auto cur_layer      = std::make_shared<TNN_NS::LayerInfo>();
        cur_layer->name     = node.output(0);
        auto type_name      = converter->TNNOpType(node, quantized_mode);
        auto layer_type     = TNN_NS::GlobalConvertLayerType(type_name);
        cur_layer->type     = layer_type;
        cur_layer->type_str = type_name;
        for (const auto& input : node.input()) {
            cur_layer->inputs.push_back(input);
        }
        for (const auto& output : node.output()) {
            cur_layer->outputs.push_back(output);
        }
        net_structure.layers.push_back(cur_layer);
        auto status =
            converter->exec(net_structure, net_resource, node, proxy_initializers, proxy_nodes, quantized_mode);
        if (status != TNN_NS::TNN_CONVERT_OK) {
            LOGE("Onnx2Tnn converter %s failed!\n", cur_layer->type_str.c_str());
            return status;
        }
        TNN_NS::ActivationType activation_function_type = converter->ActivationType(node);
        status = converter->SeparateActivation(net_structure, activation_function_type);
        if (status != TNN_NS::TNN_CONVERT_OK) {
            LOGE("Onnx2Tnn converter %s failed!\n", cur_layer->type_str.c_str());
            return status;
        }
        converter->InsertBlobs(net_structure);
    }
    return TNN_NS::TNN_CONVERT_OK;
}

bool Onnx2Tnn::ReadModel() {
    std::ifstream input_stream(this->onnx_model_path_, std::ifstream::in | std::ifstream::binary);
    if (!input_stream.is_open()) {
        LOGE("Open the %s failed\n", this->onnx_model_path_.c_str());
        return false;
    }
    google::protobuf::io::IstreamInputStream input(&input_stream);
    google::protobuf::io::CodedInputStream coded_input_stream(&input);
#if GOOGLE_PROTOBUF_VERSION >= 3002000
    coded_input_stream.SetTotalBytesLimit(INT_MAX);
#else
    coded_input_stream.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);
#endif
    bool success = this->onnx_model_->ParseFromCodedStream(&coded_input_stream);
    input_stream.close();
    return success;
}

bool Onnx2Tnn::IsQuantized() {
    // TODO
    return true;
}

}  // namespace TNN_CONVERTER
