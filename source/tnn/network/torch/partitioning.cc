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

#include "tnn/network/torch/partitioning.h"

#include <queue>

#include "tnn/network/torch/jit_util.h"
#include "tnn/network/torch/torch_op_converter.h"
#include "tnn/network/torch/shape_inference.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"

namespace TNN_NS {
namespace partitioning {

struct usage_info {
    int produce_id = -1;
    std::vector<int> torch_use_id;
    std::vector<int> tnn_use_id;
};

bool OpSupported(const torch::jit::Node* n) {
    const auto& op_type = n->kind().toQualString();

    if (conversion::GetGlobalTorchConvertMap().count(op_type) > 0) {
        auto& converter = conversion::GetGlobalTorchConvertMap()[op_type];
        if (converter->IsSupported(n))
            return true;
    }

    return false;
}

bool CheckFatalOp(const torch::jit::Node* n) {
    static std::set<std::string> fatal_op_name = {"nonzero", "zeros"};
    for (auto b : n->blocks()) {
        for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
            auto res = CheckFatalOp(*it);
            if (res)
                return res;
        }
    }

    if (fatal_op_name.count(n->kind().toUnqualString()) > 0) {
        return true;
    }

    return false;
}

bool isAllNodesSupported(const std::vector<torch::jit::Node*>& nodes) {
    for (auto node : nodes) {
        if (!OpSupported(node)) {
            return false;
        }
    }
    return true;
}

bool containTargetInputs(torch::jit::Node* n, const std::unordered_set<torch::jit::Value*>& target_inputs) {
    for (auto input : n->inputs()) {
        if (!util::isTensorOrTensorList(input) && target_inputs.count(input)) {
            return true;
        }
    }
    return false;
}

void findAllNonTensorOutputs(torch::jit::Node* n, std::vector<std::string>& non_tensor_outputs) {
    for (auto output : n->outputs()) {
        if (!util::isTensorOrTensorList(output)) {
            non_tensor_outputs.push_back(output->debugName());
        }
    }
}

bool nodeInputContainsNonTensorOutput(torch::jit::Node* n, std::vector<std::string> non_tensor_outputs) {
    for (auto input : n->inputs()) {
        if (std::find(non_tensor_outputs.begin(), non_tensor_outputs.end(), input->debugName()) \
            != non_tensor_outputs.end()) {
            return true;
        }
    }
    return false;
}

std::vector<torch::jit::Node*> getDependencyNodes(std::vector<torch::jit::Value*>& vals) {
    // use bfs to get the DAG dependency nodes for input value
    std::queue<torch::jit::Value*, std::deque<torch::jit::Value*>> q(
        std::deque<torch::jit::Value*>(vals.begin(), vals.end()));
    std::unordered_set<torch::jit::Node*> visited;
    std::vector<torch::jit::Node*> stk;
    while (!q.empty()) {
        auto cur_val = q.front();
        q.pop();
        auto node = cur_val->node();
        if (node->kind() != torch::jit::prim::Constant && !visited.count(node)) {
            stk.push_back(node);
            for (auto input : node->inputs()) {
                if (!util::isTensorOrTensorList(input)) {
                    q.push(input);
                }
            }
        }
    }
    std::reverse(stk.begin(), stk.end());
    return stk;
}

std::vector<SegmentedBlock> injectNodesForNonTensorInputs(SegmentedBlock& seg_block) {
    // reconstruct segmented_block if this block requires nonTensor input
    std::vector<torch::jit::Value*> nontensor_inputs;
    for (auto input : seg_block.raw_inputs()) {
        if (!util::isTensorOrTensorList(input)) {
            nontensor_inputs.push_back(input);
        }
    }
    std::vector<torch::jit::Node*> dependency_nodes = getDependencyNodes(nontensor_inputs);

    std::vector<SegmentedBlock> new_seg_blocks;
    // if current block is kTorch or current block is TNN and all dependent nodes are also supported, construct only
    // one new block
    if (seg_block.target() == SegmentedBlock::kTorch || isAllNodesSupported(dependency_nodes)) {
        dependency_nodes.insert(dependency_nodes.end(), seg_block.raw_nodes().begin(), seg_block.raw_nodes().end());
        new_seg_blocks.emplace_back(seg_block.target(), dependency_nodes);
    } else {
        // if current block is kTNN but the dependency nodes contain unsupported node, then we have to segment again
        std::unordered_set<torch::jit::Value*> nontensor_inputs_set(nontensor_inputs.begin(),
                                                                    nontensor_inputs.end());
        std::vector<torch::jit::Node*> tnn_nodes, pytorch_nodes;
        std::vector<std::string> nontensor_outputs;
        for (auto n : seg_block.raw_nodes()) {
            // it's a kTorch block if it uses the nonTensor input and the nonTensor input is produced in kTorch
            // block
            if (containTargetInputs(n, nontensor_inputs_set) || nodeInputContainsNonTensorOutput(n, nontensor_outputs)) {
                if (!tnn_nodes.empty()) {
                    new_seg_blocks.emplace_back(SegmentedBlock::kTNN, tnn_nodes);
                    tnn_nodes.clear();
                }
                pytorch_nodes.push_back(n);
                findAllNonTensorOutputs(n, nontensor_outputs);
            } else {
                if (!pytorch_nodes.empty()) {
                    new_seg_blocks.emplace_back(SegmentedBlock::kTorch, pytorch_nodes);
                    pytorch_nodes.clear();
                }
                tnn_nodes.push_back(n);
            }
        }
        if (!tnn_nodes.empty()) {
            new_seg_blocks.emplace_back(SegmentedBlock::kTNN, tnn_nodes);
        } else {
            new_seg_blocks.emplace_back(SegmentedBlock::kTorch, pytorch_nodes);
        }
    }
    return std::move(new_seg_blocks);
}

void resolveNonTensorInputs(PartitionedGraph& segmented_blocks, std::shared_ptr<torch::jit::Graph> g) {
    // for NonTensor inputs in TNN segments, count the usages on Torch segments and TNN segments
    std::unordered_map<torch::jit::Value*, usage_info> usage_counts;
    for (int i = segmented_blocks.size() - 1; i >= 0; --i) {
        for (auto input : segmented_blocks[i].raw_inputs()) {
            if (!util::isTensorOrTensorList(input)) {
                segmented_blocks[i].target() == SegmentedBlock::kTorch
                    ? usage_counts[input].torch_use_id.push_back(i)
                    : usage_counts[input].tnn_use_id.push_back(i);
            }
        }
        for (auto& use : usage_counts) {
            if (segmented_blocks[i].contain_raw_value(use.first)) {
                use.second.produce_id = i;
            }
        }
    }

    std::unordered_set<int> updated_old_segment_ids;
    std::map<int, int> old_to_new_segment_ids;
    for (int i=0; i<segmented_blocks.size(); i++) {
        old_to_new_segment_ids[i] = i;
    }
    for (auto& use : usage_counts) {
        auto use_info = use.second;
        // if the segment that produce this nonTensor value is kTNN but consumed in kTorch, inject nodes in the
        // first kTorch segments
        if (segmented_blocks[use_info.produce_id].target() == SegmentedBlock::kTNN &&
            !use_info.torch_use_id.empty()) {
            int first_torch_id = use_info.torch_use_id.front();
            auto tnn_nodes     = segmented_blocks[use_info.produce_id].raw_nodes();

            if (!updated_old_segment_ids.count(first_torch_id)) {
                int new_id = old_to_new_segment_ids[first_torch_id];
                auto old_block_nodes = segmented_blocks[new_id].raw_nodes();
                auto new_torch_block = injectNodesForNonTensorInputs(segmented_blocks[new_id]).front();
                auto new_block_nodes = new_torch_block.raw_nodes();
                segmented_blocks[new_id] = new_torch_block;
                updated_old_segment_ids.insert(first_torch_id);
            }
        } else {
            // KTNN segments always need to inject nodes for the nonTensor inputs
            for (int id : use_info.tnn_use_id) {
                if (!updated_old_segment_ids.count(id)) {
                    int new_id = old_to_new_segment_ids[id];
                    auto to_inject_blocks = injectNodesForNonTensorInputs(segmented_blocks[new_id]);
                    segmented_blocks.erase(segmented_blocks.begin() + new_id);
                    segmented_blocks.insert(segmented_blocks.begin() + new_id, to_inject_blocks.begin(),
                                            to_inject_blocks.end());
                    updated_old_segment_ids.insert(id);

                    if (to_inject_blocks.size()>1) {
                        int offset = to_inject_blocks.size()-1;
                        for (auto it=old_to_new_segment_ids.begin(); it!=old_to_new_segment_ids.end(); it++) {
                            if (it->first > id) {
                                it->second += offset;
                            }
                        }
                    }
                }
            }
        }
    }
    return;
}

void registerSegmentsOutputs(PartitionedGraph& segmented_blocks, std::shared_ptr<torch::jit::Graph> g) {
    // find the corresponding raw values in original global graph for this segmented block's inputs/outputs
    std::vector<torch::jit::Value*> input_values;
    for (auto& seg_block : segmented_blocks) {
        for (auto& input : seg_block.raw_inputs()) {
            input_values.push_back(input);
        }
    }

    for (auto& graph_output : g->outputs()) {
        input_values.push_back(graph_output);
    }

    // remove duplicate value and keep the order
    std::set<torch::jit::Value*> temp_set;
    auto new_end = std::remove_if(input_values.begin(), input_values.end(), [&temp_set](torch::jit::Value* value) {
        if (temp_set.find(value) != std::end(temp_set))
            return true;
        temp_set.insert(value);
        return false;
    });

    input_values.erase(new_end, input_values.end());

    // should be careful here because some in-place operations don't return any values, there is no output for this
    // kind of segment identify the output for each mini-graph by checking if any value in this graph is used later
    // we shouldn't register nonTensor output for TNN segments
    for (auto& seg_block : segmented_blocks) {
        for (auto& mini_graph_input : input_values) {
            if (std::find(seg_block.raw_inputs().begin(), seg_block.raw_inputs().end(), mini_graph_input) ==
                    seg_block.raw_inputs().end() &&
                seg_block.contain_raw_value(mini_graph_input)) {
                if (!util::isTensorOrTensorList(mini_graph_input) && seg_block.target() == SegmentedBlock::kTNN)
                    continue;
                seg_block.registerOutput(mini_graph_input);
            }
        }
        // if no output, then register the last node's output as current graph's output
        if (seg_block.raw_outputs().empty()) {
            // for Torch segments, register input as output
            if (seg_block.target() == SegmentedBlock::kTorch) {
                seg_block.registerOutput(seg_block.raw_inputs()[0]);
            } else {
                // for TNN segments, register last nonInput Tensor outputs
                for (int i = seg_block.raw_nodes().size() - 1; i >= 0; --i) {
                    for (auto node_output : seg_block.raw_nodes()[i]->outputs()) {
                        if (util::isTensorOrTensorList(node_output))
                            seg_block.registerOutput(node_output);
                    }
                    if (!seg_block.raw_outputs().empty())
                        break;
                }
            }
        }
    }

    std::for_each(segmented_blocks.begin(), segmented_blocks.end(),
                    [](SegmentedBlock& seg_block) { torch::jit::EliminateDeadCode(seg_block.g()); });

    // erase segments which still have no output
    segmented_blocks.erase(
        std::remove_if(segmented_blocks.begin(), segmented_blocks.end(),
                        [](SegmentedBlock& seg_block) { return seg_block.raw_outputs().empty(); }),
        segmented_blocks.end());

    return;
}

std::vector<SegmentedBlock> segment_graph(std::shared_ptr<torch::jit::Graph> g) {
    auto min_block_size  = 5;
    bool forced_fallback = false;

    auto nodes = g->block()->nodes();
    std::vector<SegmentedBlock> segmented_blocks;

    // segment the nodes
    std::vector<torch::jit::Node*> tnn_nodes, pytorch_nodes;
    for (const auto n : nodes) {
        if (n->kind() == torch::jit::prim::Constant) {
            continue;
        }

        if (!forced_fallback && CheckFatalOp(n)) {
            forced_fallback = true;
        }

        // This code is a special support for YOLO V5 and needs to be optimized in the future.
        auto check_list_construct = [](torch::jit::Node* node) -> bool {
            if (node->kind() != at::prim::ListConstruct) {
                return false;
            }
            if (node->next()->kind() != at::aten::view) {
                return false;
            }

            return true;
        };

        if (OpSupported(n) && !forced_fallback && !check_list_construct(n)) {
            tnn_nodes.push_back(n);
            if (tnn_nodes.size() >= min_block_size && !pytorch_nodes.empty()) {
                segmented_blocks.emplace_back(SegmentedBlock::kTorch, pytorch_nodes);
                pytorch_nodes.clear();
            }
        } else {
            if (tnn_nodes.size() >= min_block_size) {
                segmented_blocks.emplace_back(SegmentedBlock::kTNN, tnn_nodes);
            } else {
                pytorch_nodes.insert(pytorch_nodes.end(), tnn_nodes.begin(), tnn_nodes.end());
            }
            tnn_nodes.clear();
            pytorch_nodes.push_back(n);
        }
    }

    // if there is any kTorch nodes left, then either the last nodes are kTorch or last nodes are kTNN but num <
    // min_block_size
    if (!pytorch_nodes.empty()) {
        pytorch_nodes.insert(pytorch_nodes.end(), tnn_nodes.begin(), tnn_nodes.end());
        segmented_blocks.emplace_back(SegmentedBlock::kTorch, pytorch_nodes);
    } else {
        segmented_blocks.emplace_back(SegmentedBlock::kTNN, tnn_nodes);
    }

    return std::move(segmented_blocks);
}

std::vector<SegmentedBlock> Partition(torch::jit::Module& mod, std::shared_ptr<torch::jit::Graph> g, NetworkConfig& config) {
    // LOG_DEBUG(partition_info);
    // segment lowering global graph into blocks
    std::vector<SegmentedBlock> segmented_blocks = segment_graph(g);

    // resolve nonTensor inputs/outputs
    // resolveNonTensorInputs(segmented_blocks, g);

    // register input/output torch::jit::Value for segmented graphs
    registerSegmentsOutputs(segmented_blocks, g);

    // only return TNN subgraph
     
    /*
    for (auto block : segmented_blocks) {
         printf("====================== subgraph start %d ======================\n", block.target());
         // if (block.target() == SegmentedBlock::kTNN) {
         if (1) {
             std::cout << block.g()->toString(false);
         }
         printf("====================== subgraph end   %d ======================\n", block.target());
     }
    */

    segmented_blocks.erase(
        std::remove_if(segmented_blocks.begin(), segmented_blocks.end(),
                       [](SegmentedBlock& seg_block) { return seg_block.target() == SegmentedBlock::kTorch; }),
        segmented_blocks.end());

    /*
     for (auto block : segmented_blocks) {
         printf("====================== subgraph start %d ======================\n", block.target());
         // if (block.target() == SegmentedBlock::kTNN) {
         if (1) {
             std::cout << block.g()->toString(false);
         }
         printf("====================== subgraph end   %d ======================\n", block.target());
     }
*/
    return segmented_blocks;
}

}  // namespace partitioning
}  // namespace TNN_NS
