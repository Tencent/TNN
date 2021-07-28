#include "tnn/network/torch/torch_compile.h"

#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/lower_graph.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

#include "tnn/network/torch/jit_util.h"
#include "tnn/network/torch/torch_convert.h"

namespace TNN_NS {

using namespace conversion;
using namespace torch::jit;
using namespace c10;
using namespace at;

void removeUselessOps(torch::jit::Block* block, std::string self) {
    for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end; ++it) {
        for (auto b : it->blocks()) {
            removeUselessOps(b, self);
        }
        std::set<torch::jit::NodeKind> uselessKind = {
            // prime
            at::prim::Print, at::prim::RaiseException, at::prim::TimePoint, at::prim::annotate,
            // aten
            at::aten::warn,
            // at::prim::SetAttr,
            // at::aten::_set_item
        };
        if (uselessKind.count(it->kind())) {
            // if (it->kind() == at::prim::SetAttr) {
            //     if (it->input(0)->debugName().compare(self) != 0) {
            //         continue;
            //     }
            // }

            for (size_t i = 0; i < it->inputs().size();) {
                auto input = it->inputs().at(i);
                // only handling constants bc of potential side effects
                if (input->uses().size() == 1 && input->node()->kind() == at::prim::Constant) {
                    it->removeInput(i);
                    input->node()->destroy();
                } else {
                    ++i;
                }
            }
            it.destroyCurrent();
        } else if (it->kind() == at::prim::Loop) {
            if (it->outputs().empty()) {
                it.destroyCurrent();
            }
        } else if (it->kind() == at::aten::contiguous || it->kind().toUnqualString() == std::string("data")) {
            it->output()->replaceAllUsesWith(it->input(0));
            for (int i = it->inputs().size() - 1; i >= 0; i--) {
                it->removeInput(i);
            }
            it.destroyCurrent();
        }
    }
}

std::map<torch::jit::Value*, torch::jit::IValue> get_named_params(c10::ArrayRef<torch::jit::Value*> inputs,
                                                                  std::vector<torch::jit::IValue> params) {
    std::map<torch::jit::Value*, torch::jit::IValue> named_params;
    auto param_it = params.begin();
    for (auto in : inputs) {
        if (in->type()->kind() == torch::jit::TypeKind::OptionalType)
            continue;
        if (!util::isTensorOrTensorList(in) && param_it != params.end()) {
            std::cout << in->debugName() << std::endl;
            named_params[in] = *param_it;
            ++param_it;
        }
    }

    return std::move(named_params);
}

torch::jit::Value* getOrAddInputForValue(torch::jit::Value* old_value, std::shared_ptr<torch::jit::Graph>& graph,
                                         std::unordered_map<torch::jit::Value*, torch::jit::Value*>& old_to_new) {
    if (old_to_new.count(old_value) == 0) {
        auto node = old_value->node();

        if (node->kind() == torch::jit::prim::Constant) {
            auto new_const = graph->createClone(node, {nullptr});
            graph->block()->prependNode(new_const);
            return new_const->output();
        }
        auto new_value        = graph->block()->addInput();
        old_to_new[old_value] = new_value;
        new_value->copyMetadata(old_value);
        return new_value;
    } else {
        return old_to_new[old_value];
    }
}

torch::jit::Node* cloneNode(torch::jit::Node* node, std::shared_ptr<torch::jit::Graph>& graph,
                            std::unordered_map<torch::jit::Value*, torch::jit::Value*>& old_to_new) {
    auto* block = graph->block();
    auto env    = [&](torch::jit::Value* v) { return getOrAddInputForValue(v, graph, old_to_new); };

    // create node for current graph by using the metadata in node and input Values in env
    auto new_node = block->appendNode(graph->createClone(node, env));
    for (size_t i = 0; i < node->outputs().size(); ++i) {
        auto oo        = node->outputs()[i];
        auto no        = new_node->outputs()[i];
        old_to_new[oo] = no;
    }
    return new_node;
}

void AddEngineToGraph(std::shared_ptr<torch::jit::script::Module> mod, std::shared_ptr<torch::jit::Graph>& g,
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
    mod->register_attribute(engine_id, c10::getCustomClassType<c10::intrusive_ptr<runtime::TNNEngine>>(),
                            c10::IValue(std::move(engine_ptr)), false);

    // Add the module as an input into the graph
    auto self = g->addInput("self_1");
    self->setType(mod->type());

    // Start by retriveing the engine from the module attribute list
    auto engine_node = g->createGetAttr(self, engine_id);
    g->block()->appendNode(engine_node);

    // Add inputs to the graph corresponding to the number of input tensors
    // expected by the engine Also store those inputs in a vector so that they can
    // be coalesced into a single list at runtime
    std::vector<torch::jit::Value*> engine_inputs;
    for (uint64_t i = 0; i < input_names.size(); i++) {
        auto in_val = g->addInput(std::string("input_") + std::to_string(i));
        in_val->setType(c10::TensorType::get());
        engine_inputs.push_back(in_val);
    }

    // Create a node that will merge all of the input tensors into a single list
    // argument to the trt::execute_engine op Creates: prim::ListConstruct(<input
    // tensors>)
    auto input_list_node =
        g->createList(c10::TensorType::get(), torch::jit::ArrayRef<torch::jit::Value*>(engine_inputs));
    g->block()->appendNode(input_list_node);

    // Make a list of inputs to the actual trt::execute_engine op
    // Note: Ordering of list and then engine is because we can pop off the engine
    // first which contains all the metadata needed for execution
    std::vector<torch::jit::Value*> execute_node_inputs;
    execute_node_inputs.push_back(input_list_node->outputs()[0]);
    execute_node_inputs.push_back(engine_node->outputs()[0]);

    // Create the actual execution node trt::execute_engine using the assembled
    // inputs
    auto execute_node = g->create(c10::Symbol::fromQualString("tnn::execute_engine"),
                                  torch::jit::ArrayRef<torch::jit::Value*>(execute_node_inputs), 1);
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

    std::cout << g->toString() << std::endl;

    return;
}

void AddSegmentedBlockToGraph(std::shared_ptr<torch::jit::Graph>& g, partitioning::SegmentedBlock& seg,
                              std::unordered_map<torch::jit::Value*, torch::jit::Value*>& old_to_new_g) {
    // old_to_new_g contains: original global graph value => new global graph value,
    // mini_to_new_g: mini graph value -> new graph value
    std::unordered_map<torch::jit::Value*, torch::jit::Value*> mini_to_new_g;
    size_t input_idx = 0;
    if (seg.target() == partitioning::SegmentedBlock::kTNN && g->inputs().size() > 0) {
        if (g->inputs()[0]->type()->str().find("__torch__") == std::string::npos) {
            auto self = g->insertInput(0, "self_1");
            self->setType(seg.inputs()[0]->type());
        }
        mini_to_new_g[seg.inputs()[input_idx++]] = g->inputs()[0];
    }

    for (auto& raw_input : seg.raw_inputs()) {
        if (old_to_new_g.count(raw_input)) {
            mini_to_new_g[seg.inputs()[input_idx++]] = old_to_new_g[raw_input];
        }
    }

    for (const auto n : seg.nodes()) {
        cloneNode(n, g, mini_to_new_g);
    }

    // original graph value => new global graph value
    for (size_t i = 0; i < seg.raw_outputs().size(); ++i) {
        old_to_new_g[seg.raw_outputs()[i]] = mini_to_new_g[seg.outputs()[i]];
    }

    return;
}

std::shared_ptr<torch::jit::Module> CompileTorch(std::shared_ptr<torch::jit::Module> mod, InputShapesMap& input_shape) {
    std::cout << c10::toString(mod->get_method("forward").function().getSchema()) << std::endl;
    auto low_g = mod->get_method("forward").graph();
    // removeUselessOps(low_g->block(), low_g->inputs()[0]->debugName());
    // return mod;
    std::cout << low_g->toString(false) << std::endl;

    // auto graph_and_ivalues = torch::jit::tnn::LowerGraph(*low_g, mod->_ivalue());
    // auto graph_and_ivalues = torch::jit::LowerGraph(*low_g, mod->_ivalue());
    // auto g = graph_and_ivalues.first;
    auto g = low_g;

    std::unordered_map<torch::jit::Value*, torch::jit::Value*> old_to_new_g;

    for (auto input : g->inputs()) {
        std::cout << input->debugName() << " | " << input->type()->repr_str() << std::endl;
    }
    // auto named_params = get_named_params(g->inputs(), graph_and_ivalues.second);

    auto seg_blocks = partitioning::Partition(g, input_shape);

    int subgraph_cnt = 0;
    for (auto& block : seg_blocks) {
        std::string cur_block_target = block.target() == partitioning::SegmentedBlock::kTNN ? "TNN" : "Torch";
        std::ostringstream tnn_engine_id;
        tnn_engine_id << reinterpret_cast<const int*>(&block);
        if (block.target() == partitioning::SegmentedBlock::kTNN) {
            // if (subgraph_cnt >= 4) {
            //     block.change_target(partitioning::SegmentedBlock::kTorch);
            //     continue;
            // }
            auto engine_ptr = conversion::ConvertBlockToInstance(block);
            auto temp_g     = std::make_shared<torch::jit::Graph>();
            AddEngineToGraph(mod, temp_g, engine_ptr, tnn_engine_id.str(), true);
            // std::cout << block.g()->toString() << std::endl;
            // std::cout << temp_g->toString() << std::endl;

            std::vector<torch::jit::Value*> block_real_inputs;
            block_real_inputs.push_back(low_g->inputs()[0]);
            for (auto input : block.raw_inputs()) {
                if (old_to_new_g.count(input) == 0) {
                    block_real_inputs.push_back(input);
                } else {
                    block_real_inputs.push_back(old_to_new_g[input]);
                }
            }
            for (auto input : block_real_inputs) {
                std::cout << input->debugName() << " | " << input->owningGraph() << std::endl;
            }

            WithInsertPoint insert_point(block.raw_outputs()[0]->node());
            auto new_outputs = torch::jit::insertGraph(*low_g, *temp_g, block_real_inputs);

            int out_idx = 0;
            for (auto output : block.raw_outputs()) {
                // std::cout << output->debugName() << " | " << output->owningGraph() << " | " <<
                // new_outputs[out_idx]->debugName() << std::endl;
                output->replaceAllUsesWith(new_outputs[out_idx]);
                old_to_new_g[output] = new_outputs[out_idx++];
            }

            // std::cout << low_g->toString() << std::endl;

            subgraph_cnt++;

            block.update_graph(temp_g);
            // AddSegmentedBlockToGraph(new_g, block, old_to_new_g);
        } else {
            // AddSegmentedBlockToGraph(new_g, block, old_to_new_g);
        }
    }

    for (auto& block : seg_blocks) {
        if (block.target() == partitioning::SegmentedBlock::kTNN) {
            for (auto n : block.raw_nodes()) {
                n->removeAllInputs();
            }
            for (auto n : block.raw_nodes()) {
                n->destroy();
            }
        }
    }

    std::cout << "============================= the final graph ===========================" << std::endl;
    std::cout << low_g->toString() << std::endl;

    return mod;
}

}  // namespace TNN_NS