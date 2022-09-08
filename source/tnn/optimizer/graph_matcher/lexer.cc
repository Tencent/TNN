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

#include "tnn/optimizer/graph_matcher/lexer.h"

#include "tnn/core/macro.h"
#include "tnn/core/layer_type.h"

namespace TNN_NS {

bool startsWith(const std::string &input, const std::string &pattern) {
    return input.substr(0, pattern.length()) == pattern;
}

std::string layerTypeName(LayerType type) {
    for(auto it : GetGlobalLayerTypeMap()) {
        if (it.second == type && 
            !startsWith(it.first, "Quantized") && 
            !startsWith(it.first, "DynamicRangeQuantized") ) 
        {
            return it.first;
        }
    }
    return "Unknown";
}

std::string tokenName(int kind) {
    if (kind < 256)
        return std::string(1, kind);

    switch (kind) {
#define TK_NAME(tok, str, _) \
    case tok:                      \
    return str;
        TNN_ALL_TOKEN_KINDS(TK_NAME)
#undef TK_NAME
        default:
        char _ss[100]; 
        snprintf(_ss, 100, "Unknown kind:%d", kind);
        throw std::runtime_error(_ss);
    } 
}

void expect(const Token &tk, const std::vector<int> kinds) {
    if (std::find(kinds.begin(), kinds.end(), tk.kind) == kinds.end()) {
        std::stringstream ss;
        ss << "Expected token types : ";
        for(auto kind: kinds)
            ss << tokenName(kind) << " ";
        ss << ", but got " << tk.name() << ":\n";
        tk.str.highlight(ss);
        throw std::runtime_error(ss.str());
    }
}
void expect(const Token &tk, const int kind) {
    if (tk.kind != kind) {
        std::stringstream ss;
        ss << "Expected token type " << tokenName(kind);
        ss << " but got " << tk.name() << ":\n";
        tk.str.highlight(ss);
        throw std::runtime_error(ss.str());
    }
}

void unexpect(const Token &tk) {
    std::stringstream ss;
    ss << "Unexpected " << tk.name() << ":\n";
    tk.str.highlight(ss);
    throw std::runtime_error(ss.str());
}

void reportError(const std::string &msg, const Token &tok) {
    std::stringstream ss;
    ss << "Error: " << msg << ", correspoding source is:\n";
    tok.str.highlight(ss);
    throw std::runtime_error(ss.str());
}


} // namespace tnn