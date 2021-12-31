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

#include "tnn/network/torch/torch_compile.h"

#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/torch.h>

#include "tnn/network/torch/jit_util.h"
#include "tnn/network/torch/partitioning.h"
#include "tnn/network/torch/torch_convert.h"
#include "tnn/network/torch/torch_optimize.h"

#include "tnn/utils/blob_dump_utils.h"
#include <c10/cuda/CUDACachingAllocator.h>

namespace TNN_NS {

using namespace conversion;
using namespace torch::jit;

void AddEngineToGraph(torch::jit::script::Module& mod, std::shared_ptr<torch::jit::Graph> &g,
                      c10::intrusive_ptr<runtime::TNNEngine> engine_ptr, std::string engine_id = "",
                      bool fallback = false) {
    // Get required metadata about the engine out
    // BlobMap input_blobs;
    // BlobMap output_blobs;
    // engine_ptr->instance_->GetAllInputBlobs(input_blobs);
    // engine_ptr->instance_->GetAllOutputBlobs(output_blobs);
    auto input_names  = engine_ptr->input_names;
    auto output_names = engine_ptr->output_names;

    //..
    // Add the engine as an attribute of the module, this will let the engine be
    // serialized and deserialized
    mod.register_attribute(engine_id, c10::getCustomClassType<c10::intrusive_ptr<runtime::TNNEngine>>(),
                            c10::IValue(std::move(engine_ptr)), false);

    // Add the module as an input into the graph
    auto self = g->addInput("self_1");
    self->setType(mod.type());

    // Start by retriveing the engine from the module attribute list
    auto engine_node = g->createGetAttr(self, engine_id);
    g->block()->appendNode(engine_node);

    // Add inputs to the graph corresponding to the number of input tensors
    // expected by the engine Also store those inputs in a vector so that they can
    // be coalesced into a single list at runtime
    std::vector<torch::jit::Value *> engine_inputs;
    for (uint64_t i = 0; i < input_names.size(); i++) {
        auto in_val = g->addInput(std::string("input_") + std::to_string(i));
        in_val->setType(c10::TensorType::get());
        engine_inputs.push_back(in_val);
    }

    // Create a node that will merge all of the input tensors into a single list
    // argument to the trt::execute_engine op Creates: prim::ListConstruct(<input
    // tensors>)
    auto input_list_node =
        g->createList(c10::TensorType::get(), torch::jit::ArrayRef<torch::jit::Value *>(engine_inputs));
    g->block()->appendNode(input_list_node);

    // Make a list of inputs to the actual trt::execute_engine op
    // Note: Ordering of list and then engine is because we can pop off the engine
    // first which contains all the metadata needed for execution
    std::vector<torch::jit::Value *> execute_node_inputs;
    execute_node_inputs.push_back(input_list_node->outputs()[0]);
    execute_node_inputs.push_back(engine_node->outputs()[0]);

    // Create the actual execution node trt::execute_engine using the assembled
    // inputs
    auto execute_node = g->create(c10::Symbol::fromQualString("tnn::execute_engine"),
                                  torch::jit::ArrayRef<torch::jit::Value *>(execute_node_inputs), 1);
    g->block()->appendNode(execute_node);
    execute_node->outputs()[0]->setType(c10::ListType::ofTensors());

    // Create a node to unpack the list into seperate tensors, in the case of
    // there being only one tensor, the tensor will be returned, otherwise they
    // are returned as a tuple of tensors. Creates: prim::ListUnpack(<engine
    // output>)
    auto unpack_node = g->createListUnpack(execute_node->outputs()[0], output_names.size());
    g->block()->appendNode(unpack_node);

    // If there are multiple output tensors from TensorRT we wrap them in a tuple
    // to return, convert to tuple only when we only have 1 segmented graph
    if (!fallback && unpack_node->outputs().size() > 1) {
        // Creates prim::TupleConstruct(<output tensors>) using outputs of the
        // unpack node
        auto return_tuple_node = g->createTuple(unpack_node->outputs());
        g->block()->appendNode(return_tuple_node);
        // Set the output as the produced tuple
        g->registerOutput(return_tuple_node->outputs()[0]);
    } else {
        // if fallback is enabled, multiple outputs will be registered
        for (size_t i = 0; i < unpack_node->outputs().size(); ++i) {
            g->registerOutput(unpack_node->outputs()[i]);
        }
    }

    return;
}

void RegisterNodeToOutput(std::shared_ptr<torch::jit::Module> &mod, const std::vector<torch::jit::Node *> &nodes) {
    auto copy_g = mod->get_method("forward").graph();

    std::vector<torch::jit::Value *> out_vec;
    for (auto &output : copy_g->outputs()) {
        out_vec.push_back(output);
    }

    for (auto node : nodes) {
        for (auto &output : node->outputs()) {
            out_vec.push_back(output);
        }
        // copy_g->registerOutput(output);
    }
    c10::ArrayRef<torch::jit::Value *> new_outputs(out_vec);
    auto new_output_node = copy_g->appendNode(copy_g->createTuple(new_outputs));
    for (int idx = copy_g->outputs().size() - 1; idx >= 0; --idx) {
        copy_g->eraseOutput(idx);
    }
    copy_g->registerOutput(new_output_node->outputs()[0]);

    auto &cur_method = mod->get_method("forward").function();
    auto schema      = util::GenerateGraphSchema(cur_method.name(), copy_g);
    cur_method.setSchema(schema);
}

torch::jit::Module CompileTorch(torch::jit::Module &mod, InputShapesMap &min_input_shape,
                                InputShapesMap &max_input_shape, InputDataTypeMap &input_type,
                                NetworkConfig &config, std::string forward_func_name) {
    if (config.precision == PRECISION_LOW ) {
        mod.to(torch::kHalf);
    }

    TorchOptPass(mod);

    // std::cout << c10::toString(mod.get_method("forward").function().getSchema()) << std::endl;
    auto g = mod.get_method(forward_func_name).graph();
    // std::cout << g->toString(false) << std::endl;

    std::unordered_map<torch::jit::Value *, torch::jit::Value *> old_to_new_g;

    // for (auto input : g->inputs()) {
    //     std::cout << input->debugName() << " | " << input->type()->repr_str() << std::endl;
    // }

    try {
        auto seg_blocks = partitioning::Partition(mod, g, config);

        // run shape infer and combine to blocks
        if (min_input_shape.size() && max_input_shape.size() && min_input_shape.size() == max_input_shape.size()) {
	    //fix clone memory leak
	    std::stringstream save_stream(std::ios_base::binary | std::ios_base::in | std::ios_base::out);
	    mod.save(save_stream);
	    save_stream.seekg(0);
	    c10::Device device(c10::kCPU);
            ConvertToTorchDevice(device, config.device_type, config.device_id);
	    auto shape_mod = torch::jit::freeze(torch::jit::load(save_stream, device));
            auto shape_seg = partitioning::Partition(shape_mod, shape_mod.get_method(forward_func_name).graph(), config);
            std::vector<BlobDesc> subgraph_min_input_info;
            std::vector<BlobDesc> subgraph_max_input_info;
            //// input type & shape will be used for random input generation, then subgraph input info can be infered out
            // InputDataTypeMap input_type;

            partitioning::runShapeInfer(shape_mod, shape_seg, min_input_shape, input_type, config, subgraph_min_input_info);
            partitioning::runShapeInfer(shape_mod, shape_seg, max_input_shape, input_type, config, subgraph_max_input_info);
	    
            int input_idx = 0;
            for (auto &block : seg_blocks) {
                std::vector<DimsVector> min_shape;
                std::vector<DimsVector> max_shape;
                std::vector<DataType> in_type;
                for (auto &input : block.raw_inputs()) {
                    min_shape.push_back(subgraph_min_input_info[input_idx].dims);
                    max_shape.push_back(subgraph_max_input_info[input_idx].dims);
                    in_type.push_back(subgraph_max_input_info[input_idx].data_type);
                    input_idx++;
                }
                block.register_min_inshape(min_shape);
                block.register_max_inshape(max_shape);
                block.register_intype(in_type);
            }

            if (config.device_type == DEVICE_CUDA) {
                // release cached cuda memory
                c10::cuda::CUDACachingAllocator::emptyCache();
            }
        }
    #if (DUMP_INPUT_BLOB || DUMP_OUTPUT_BLOB)
        {
            std::vector<torch::jit::Node *> reg_outputs;
            for (auto &block : seg_blocks) {
                if (block.target() == partitioning::SegmentedBlock::kTNN) {
                    for (auto &node : block.raw_nodes()) {
                        reg_outputs.push_back(node);
                    }
                }
            }
            RegisterNodeToOutput(mod, reg_outputs);
            return;
        }
    #endif

        int block_idx = 0;
        int block_stop_idx = INT_MAX;
        for (auto &block : seg_blocks) {
            try {
                std::ostringstream tnn_engine_id;
                tnn_engine_id << reinterpret_cast<const int *>(&block);
                auto engine_ptr = conversion::ConvertBlockToInstance(block, config);
                auto temp_g     = std::make_shared<torch::jit::Graph>();
                AddEngineToGraph(mod, temp_g, engine_ptr, tnn_engine_id.str(), true);
                // std::cout << block.g()->toString() << std::endl;
                // std::cout << temp_g->toString() << std::endl;

                std::vector<torch::jit::Value *> block_real_inputs;
                block_real_inputs.push_back(g->inputs()[0]);
                for (auto input : block.raw_inputs()) {
                    if (old_to_new_g.count(input) == 0) {
                        block_real_inputs.push_back(input);
                    } else {
                        block_real_inputs.push_back(old_to_new_g[input]);
                    }
                }

                WithInsertPoint insert_point(block.raw_outputs()[0]->node());
                auto new_outputs = torch::jit::insertGraph(*g, *temp_g, block_real_inputs);

                int out_idx = 0;
                for (auto output : block.raw_outputs()) {
                    output->replaceAllUsesWith(new_outputs[out_idx]);
                    old_to_new_g[output] = new_outputs[out_idx++];
                }

                block.update_graph(temp_g);
            } catch (std::exception& e) {
                block_stop_idx = block_idx;
                // std::cout << "exception block " << block_stop_idx << std::endl;
                std::cout << e.what() << std::endl;
                break;
            }
            block_idx++;
        }

        block_idx = 0;
        for (auto &block : seg_blocks) {
            if (block_idx < block_stop_idx) {
                for (auto n : block.raw_nodes()) {
                    n->removeAllInputs();
                }
                for (auto n : block.raw_nodes()) {
                    // node may be used in different block, destory in the last used block
                    if (std::find_if(n->outputs().begin(), n->outputs().end(), [](auto output) {
                            return output->uses().size() > 0;
                        }) != n->outputs().end()) {
                        continue;
                    }

                    n->destroy();
                }
            } else {
                break;
            }
            block_idx++;
        }
    } catch (std::exception &e) {
        std::cout << "compile exception:" << e.what() << std::endl;
        return mod;
    }

    // remove constant nodes which has been convert to tnn netresource
    torch::jit::EliminateDeadCode(g);

    // std::cout << "============================= the final graph ===========================" << std::endl;
    // std::cout << g->toString() << std::endl;

    if (config.device_type == DEVICE_CUDA) {
        // release cached cuda memory
        c10::cuda::CUDACachingAllocator::emptyCache();
    }

    return mod;
}

}  // namespace TNN_NS
