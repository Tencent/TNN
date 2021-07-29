
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

#include "onnx2tnn.h"

int Onnx2TNN::TransferInputName(onnx::GraphProto* mutable_graph) {
    // for input name
    // images_0 <- images:0
    std::set<std::string> initializers;
    const int initializer_count = mutable_graph->initializer_size();
    for (int i = 0; i < initializer_count; i++) {
        const auto& initializer_name = mutable_graph->initializer(i).name();
        initializers.insert(initializer_name);
    }

    std::map<std::string, std::string> hack_names_map;
    for (int i = 0; i < mutable_graph->input_size(); ++i) {
        const std::string& name = mutable_graph->mutable_input(i)->name();
        if (name.find(':') != std::string::npos && initializers.find(name) == initializers.end()) {
            // graph input's name has special character ':'
            std::string hack_name = name;
            std::replace(hack_name.begin(), hack_name.end(), ':', '_');
            hack_names_map[name] = hack_name;
            mutable_graph->mutable_input(i)->set_name(hack_name.c_str());
        }
    }
    if (hack_names_map.empty()) {
        return 0;
    }
    int node_count = mutable_graph->node_size();

    for (int j = 0; j < node_count; j++) {
        auto node = mutable_graph->mutable_node(j);

        do {
            for (int k = 0; k < node->input_size(); ++k) {
                std::string* node_input_name = node->mutable_input(k);
                if (hack_names_map.find(*node_input_name) !=
                    hack_names_map.end()) {
                    *node_input_name = hack_names_map[*node_input_name];
                }
            }
        } while (0);
    }
    return 0;
}
