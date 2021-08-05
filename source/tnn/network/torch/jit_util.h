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

#ifndef TNN_SOURCE_NETWORK_TNNTORCH_TNNTORCH_JIT_UTIL_H
#define TNN_SOURCE_NETWORK_TNNTORCH_TNNTORCH_JIT_UTIL_H

#include <sstream>
#include <string>

#include "torch/csrc/jit/ir/ir.h"

namespace TNN_NS {
namespace util {

inline bool isTensorOrTensorList(torch::jit::Value* val) {
    return val->type()->isSubtypeOf(torch::jit::TensorType::get()) ||
            val->type()->isSubtypeOf(torch::jit::ListType::ofTensors());
}

inline std::string node_info(const torch::jit::Node* n) {
    std::stringstream ss;
    ss << *n;
    std::string node_info = ss.str();
    node_info.erase(std::remove(node_info.begin(), node_info.end(), '\n'), node_info.end());
    return node_info;
}

inline std::string schema_info(const torch::jit::FunctionSchema* s) {
    std::stringstream ss;
    ss << *s;
    std::string schema_info = ss.str();
    schema_info.erase(std::remove(schema_info.begin(), schema_info.end(), '\n'), schema_info.end());
    return schema_info;
}

inline std::vector<int64_t> toVec(c10::IntArrayRef a) {
    std::vector<int64_t> arr;
    for (auto i : a) {
        arr.push_back(i);
    }
    return arr;
}

inline std::vector<int> toDims(c10::IntArrayRef l) {
    std::vector<int> dims;
    for (size_t i = 0; i < l.size(); i++) {
        dims.push_back(l[i]);
    }
    return dims;
}

inline c10::FunctionSchema GenerateGraphSchema(std::string method_name, std::shared_ptr<torch::jit::Graph>& g) {
    std::vector<c10::Argument> args;
    for (auto in : g->inputs()) {
        if (in->type()->kind() == torch::jit::TypeKind::OptionalType) {
            auto ival = torch::jit::IValue();
            args.push_back(c10::Argument(in->debugName(), in->type(), c10::nullopt, ival));
        } else {
            args.push_back(c10::Argument(in->debugName(), in->type()));
        }
    }

    std::vector<c10::Argument> returns;
    for (auto out : g->outputs()) {
        returns.push_back(c10::Argument(out->debugName(), out->type()));
    }

    return c10::FunctionSchema(method_name, method_name, args, returns);
}

}  // namespace util
}  // namespace TNN_NS

#endif
