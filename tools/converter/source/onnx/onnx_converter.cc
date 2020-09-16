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

TNN_NS::Status Onnx2Tnn::Conveter2Tnn(tnn::NetStructure net_structure, tnn::NetResource net_resource) {
    if (!ReadModel()) {
        return TNN_NS::TNNERR_CONVERT_INVALID_MODEL;
    }
    const auto& onnx_graph = onnx_model_->graph();
    std::shared_ptr<OnnxProxyGraph> onnx_proxy_graph(new OnnxProxyGraph(&onnx_graph));
    const auto& proxy_initializers      = onnx_proxy_graph->proxy_initializers_map_;
    const auto& proxy_inputs            = onnx_proxy_graph->proxy_inputs_map_;
    const auto& proxy_outputs           = onnx_proxy_graph->proxy_outputs_map_;
    const auto& proxy_nodes             = onnx_proxy_graph->proxy_nodes_map_;
    const auto& constant_node_to_delete = onnx_proxy_graph->constant_node_to_delete_;

    bool quantized_mode = IsQuantized();
    // convert onnx graph input
    TNN_NS::InputShapesMap input_shapes_map = net_structure.inputs_shape_map;
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
            //            LOGE("The onnx have support input shape\n");
            //            return TNN_NS::TNNERR_CONVERT_INVALID_MODEL;
            continue;
        }
        if (input_shapes_map.find(input_name) == input_shapes_map.end()) {
            input_shapes_map[input_name] = dims_vector;
        }
    }
    // convert onnx graph output
    auto& outputs = net_structure.outputs;
    for (const auto iter : proxy_outputs) {
        auto output_name = iter.first;
        if (outputs.find(output_name) == outputs.end()) {
            outputs.insert(output_name);
        }
    }

    // convert onnx nodes
    const auto node_size = onnx_graph.node_size();
    for (int i = 0; i < node_size; ++i) {
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
    coded_input_stream.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);
    bool success = this->onnx_model_->ParseFromCodedStream(&coded_input_stream);
    input_stream.close();
    return success;
}

bool Onnx2Tnn::IsQuantized() {
    // TODO
    return true;
}

}  // namespace TNN_CONVERTER