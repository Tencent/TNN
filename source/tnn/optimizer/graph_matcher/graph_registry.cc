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

#include "tnn/optimizer/graph_matcher/graph_registry.h"

#include <stack>
#include <list>
#include <set>

#include "tnn/core/macro.h"
#include "tnn/core/status.h"
#include "tnn/optimizer/graph_matcher/ir.h"
#include "tnn/optimizer/graph_matcher/logger.h"
#include "tnn/optimizer/graph_matcher/lexer.h"

namespace TNN_NS {

Status CheckToken(std::string name, Tokenizer* t) {
    Token tk;
    SubStr s(name);
    bool matched = t->match(s, true, &tk);
    if (!matched) {
        ERRORV("specified name:%s is invalid token", msg, name.c_str());
        return Status(TNNERR_COMMON_ERROR, msg);
    }
    if (tk.kind != TK_IDENT) {
        ERRORV("specified name:%s alread used as %s by Tokenizer:%p.", msg, name.c_str(), tokenName(tk.kind).c_str(), t);
        return Status(TNNERR_COMMON_ERROR, msg);
    }
    return TNN_OK;
}

    
Status GraphRegistry::registerTokenizer(Tokenizer * ptr) {
    consumers.push_back(ptr);
    // DEBUG("Register tokenizer : %p, now has:%lu", ptr, consumers.size());
    for(auto &pair : graphs) {
        RETURN_ON_FAIL(CheckToken(pair.first, ptr));
        ptr->onNewToken(pair.first, TK_GRAPH_FUNCTION);
    }
    return TNN_OK;
}

Status GraphRegistry::unRegisterTokenizer(Tokenizer * ptr) {
    if (std::find(consumers.begin(), consumers.end(), ptr) == consumers.end()) {
        ERRORV("unRegister tokenizer got unknown ptr:%p", msg, ptr);
        return Status(TNNERR_COMMON_ERROR, msg);
    }

    for(auto &pair : graphs) {
        ptr->onDelete(pair.first);
    }
    consumers.remove(ptr);
    return TNN_OK;
}

Status GraphRegistry::registerGraph(const std::string name, std::shared_ptr<const Graph> graph) {
    if (graphs.count(name) > 0) {
        ERRORV("GraphReigster already registered a graph with name:%s", msg, name.c_str());
        return Status(TNNERR_COMMON_ERROR, msg);
    }
    DEBUG("GraphRegistry registering a Graph with name:%s", name.c_str());

    graphs[name] = graph;
    for(auto p : consumers) {
        RETURN_ON_FAIL(CheckToken(name, p));
        p->onNewToken(name, TK_GRAPH_FUNCTION);
    }

    return TNN_OK;
}

std::shared_ptr<const Graph>  GraphRegistry::queryGraphByName(const std::string name) {
    if (graphs.count(name) > 0) {
        return graphs.at(name);
    }
    return nullptr;
}

} // namespace TNN_NS
