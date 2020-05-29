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

#include <math.h>
#include "half_utils.h"
#include "objseri.h"
#include "onnx2tnn.h"

int Onnx2TNN::FuseDepthToSpace(onnx::GraphProto* mutable_graph,
                                    std::vector<IndexNode> & index_nodes,
                                    std::map<std::string, onnx::TensorProto>& weights,
                                    std::map<std::string, int>& node_reference,
                                    std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;

        // reorg DepthToSpace <= Reshape  临时Hack，将当前Opset中不支持的 DepthToSpace 转为了reshape, 这里识别并转换回去
        /*
        do {
            auto next_op_type = mutable_graph->mutable_node(i)->op_type();
            if (next_op_type == "Reshape" && i < node_count) {

                onnx::NodeProto* cur_node = mutable_graph->mutable_node(i);


                // 找到输入节点
                int input_node_id = node_name_to_node_id[cur_node->input(0)];
                onnx::NodeProto* node_input  = mutable_graph->mutable_node(input_node_id);

                // printf("found reshape %s input: %s of type: %s\n", cur_node->output(0).c_str(), node_input->output(0).c_str(),
                //     node_input->op_type().c_str());

                if (node_input->op_type()   == "Reshape") {
                    printf("found Depth2space %s input size:%d\n", cur_node->output(0).c_str(), cur_node->input_size());
                } else {
                    break;
                }

                cur_node->set_op_type("DepthToSpace");

            }
        } while (0);
        */
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
