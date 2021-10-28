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

#include "tnn/network/torch/shape_inference.h"

#include "tnn/network/torch/jit_util.h"
#include "tnn/network/torch/torch_utils.h"
#include "tnn/network/torch/torch_tensor.h"
#include "tnn/network/torch/torch_types.h"

#include "torch/csrc/jit/api/module.h"
#include <vector>

namespace TNN_NS {
namespace partitioning {

void genRandomInputs(std::shared_ptr<torch::jit::Graph> graph, InputShapesMap &input_shape,
                        std::vector<std::shared_ptr<Blob>> &blobs,
                        std::vector<torch::jit::IValue> &jit_inputs_ivalues, NetworkConfig& config) {
    std::set<c10::TypeKind> supported_kinds = {
        c10::TypeKind::TensorType,
        c10::TypeKind::TupleType,
        c10::TypeKind::ListType,
        c10::TypeKind::DictType,
    };

    std::vector<torch::jit::Value *> inputs;
    for (auto &input : graph->block()->inputs()) {
        c10::TypeKind kind = input->type()->kind();
        if (supported_kinds.find(kind) != supported_kinds.end()) {
            inputs.push_back(input);
        }
    }

    // Create Ivalues from types
    jit_inputs_ivalues.resize(inputs.size());
    for (int i = 0; i < inputs.size(); i++) {
        CreateIValueFromTypePtr(jit_inputs_ivalues[i], inputs[i]->type());
    }

    // Create Blob and tensor
    for (auto &input : input_shape) {
        // create blob from input_shape
        BlobDesc blob_desc;
        blob_desc.device_type = config.device_type;
        blob_desc.dims        = input.second;
        auto blob             = std::make_shared<Blob>(blob_desc, true);
        // extend lifetime util shape infer ends
        blobs.push_back(blob);

        int id = JitTypeMatcher::idFromName(input.first);

        auto router = IValueRouter::create(inputs[id]->type(), input.first); 
        std::shared_ptr<at::Tensor> tensor;
        CreateTensorByBlob(tensor, blob.get());
        router->attach(jit_inputs_ivalues[id], tensor);
    }
}

void runShapeInfer(torch::jit::Module& mod, std::vector<SegmentedBlock> &segmented_blocks,
                   InputShapesMap &input_shape, NetworkConfig& config) {
    auto graph = mod.get_method("forward").graph();
    std::vector<torch::jit::Value *> new_vec;
    for (auto &block : segmented_blocks) {
        if (block.target() == partitioning::SegmentedBlock::kTNN) {
            for (auto &input : block.raw_inputs()) {
                new_vec.push_back(input);
            }
        }
    }

    // preserve old graph output
    auto old_output = graph->outputs()[0];

    // create new node for shape inference
    c10::ArrayRef<torch::jit::Value *> new_outputs(new_vec);
    auto new_return_node = graph->appendNode(graph->createTuple(new_outputs));
   
    graph->eraseOutput(0);
    graph->registerOutput(new_return_node->outputs()[0]);

    // construct a ts module
    // auto copy_g = graph->copy();
    // torch::jit::script::Module cur_mod(c10::QualifiedName("module"));
    // auto self = copy_g->insertInput(0, "self_1");
    // self->setType(cur_mod.type());
    // auto cur_method = cur_mod._ivalue()->compilation_unit()->create_function(c10::QualifiedName("forward"), copy_g);
    // auto schema     = util::GenerateGraphSchema(cur_method->name(), copy_g);
    // cur_mod.type()->addMethod(cur_method);
    // cur_method->setSchema(schema);

    // forward with random inputs
    std::vector<torch::jit::IValue> jit_inputs_ivalues;
    std::vector<std::shared_ptr<Blob>> blobs;
    genRandomInputs(graph, input_shape, blobs, jit_inputs_ivalues, config);
    torch::jit::IValue jit_results_ivalues = mod.forward(jit_inputs_ivalues);

    auto print_output_shape = [&](torch::jit::IValue &output) {
        // get result tensor shape
        if (output.isTuple()) {
            auto results = output.toTuple()->elements();
            for (auto &r : results) {
                auto shape = r.toTensor().sizes();
                auto dims = util::toDims(shape);
                std::cout << dims << std::endl;
            }
        } else {
            auto result = output.toTensor();
            auto shape = result.sizes();
            auto dims = util::toDims(shape);
            std::cout << dims << std::endl;
        }
    };

    print_output_shape(jit_results_ivalues);

    // restore old graph output
    graph->eraseOutput(0);
    graph->registerOutput(old_output);

    new_return_node->removeAllInputs();
    new_return_node->destroy();
}

}  // namespace partitioning
}  // namespace TNN_NS
