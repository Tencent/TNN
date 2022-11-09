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

#include "torch/torch_converter.h"

#include <utility>

#include "tnn/core/common.h"
#include "torch/torch_base_converter.h"
#include "torch/torch_optimize.h"
#include "torch/torch_utils.h"

namespace TNN_CONVERTER {

Torch2Tnn::Torch2Tnn(const std::string& model_path) {
    torch_model_path_ = model_path;
}
Torch2Tnn::Torch2Tnn(const std::string& model_path, const std::string& onnx_path) {
    // do nothing for now
}
Torch2Tnn::Torch2Tnn(const std::string& mode_path, const std::string& model_name, const std::string& onnx_path) {
    // do nothing for now
}
TNN_NS::Status Torch2Tnn::Convert2Tnn(tnn::NetStructure& net_structure, tnn::NetResource& net_resource) {
    ReadModel(this->torch_model_path_);
    this->torch_graph_ = torch::jit::torchOptPass(this->torch_model_);

    // parse input
    TNN_NS::InputShapesMap& input_shapes_map      = net_structure.inputs_shape_map;
    TNN_NS::InputDataTypeMap& input_data_type_map = net_structure.input_data_type_map;
    std::set<std::string>& graph_outputs          = net_structure.outputs;
    bool quantized_mode                           = IsQuantized();
    for (const auto& input : this->torch_graph_->inputs()) {
        const auto& type = input->type()->cast<at::TensorType>();
        if (!type) {
            continue;
        }
        std::string input_name     = input->debugName();
        const auto input_data_type = TorchDataType2TnnDataType(type->scalarType().value_or(at::ScalarType::Float));
        // TODO: input shape must be set by people
        TNN_NS::DimsVector input_dims = {1, 3, 128, 128};
        if (input_shapes_map.find(input_name) == input_shapes_map.end()) {
            input_shapes_map[input_name]    = input_dims;
            input_data_type_map[input_name] = input_data_type;
        }
    }
    // parse output
    for (const auto& output : this->torch_graph_->outputs()) {
        const auto& output_name = output->debugName();
        graph_outputs.insert(output_name);
    }
    // parse node
    for (const auto& node : this->torch_graph_->nodes()) {
        std::string node_name = node->output(0)->debugName();
        const auto& kind      = node->kind();
        bool is_output_node   = false;
        for (const auto& output : node->outputs()) {
            is_output_node |=
                std::find(graph_outputs.begin(), graph_outputs.end(), output->debugName()) != graph_outputs.end();
        }
        if (!is_output_node && kind.is_prim() && DealPrime(node)) {
            continue;
        }
        std::string op_type = node->kind().toQualString();
        auto* converter     = TorchConverterManager::get()->serach(op_type);
        if (converter == nullptr) {
            LOGE("The TorchConverter do not support layer:%s\n", node_name.c_str());
            LOGE("The unsupported operator type is %s\n", op_type.c_str());
            return TNN_NS::TNNERR_CONVERT_UNSUPPORT_LAYER;
        }
       auto status =  converter->exec(net_structure, net_resource, node, quantized_mode);
       if (status != TNN_NS::TNN_CONVERT_OK) {
            LOGE("Torch2Tnn converter %s failed!\n", node_name.c_str());
            return status;
       }
       converter->InsertBlobs(net_structure);
    }
    return TNN_NS::TNN_CONVERT_OK;
}
void Torch2Tnn::ReadModel(std::string torch_model_path) {
    c10::Device device("cpu");
    this->torch_model_ = torch::jit::load(torch_model_path, device);
}
bool Torch2Tnn::IsQuantized() {
    return true;
}

}  // namespace TNN_CONVERTER
