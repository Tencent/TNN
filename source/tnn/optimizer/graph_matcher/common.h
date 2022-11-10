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

#ifndef TNN_SOURCE_TNN_NET_OPTIMIZER_GRAPH_MATCHER_COMMON_H_
#define TNN_SOURCE_TNN_NET_OPTIMIZER_GRAPH_MATCHER_COMMON_H_

#include <memory>
#include <string>
#include <tuple>
#include <map>
#include <set>
#include <ostream>
#include <vector>

#include "tnn/core/macro.h"
#include "tnn/core/status.h"

#define RAISE_ON_ERROR(status)                                  \
    do {                                                        \
        auto _status = status;                                  \
        if ((_status) != TNN_OK) {                              \
            throw std::runtime_error(                           \
                std::string(__PRETTY_FUNCTION__) +              \
                std::string(" : status ") +                     \
                _status.description());                         \
        }                                                       \
    } while (0)

#define NAMES(...) __VA_ARGS__

#define CREATE_NODE(var_name, g, LAYER_TYPE, ins, outs)         \
    auto var_name = std::shared_ptr<Node>(nullptr);             \
    do {                                                        \
        auto status = g->createNode(LAYER_TYPE, ins, outs);     \
        if (status != TNN_OK) {                                 \
            ERROR("create node of type %d failed, msg:%s \n",   \
                 LAYER_TYPE, status.description().c_str());     \
            return nullptr;                                     \
        }                                                       \
        var_name = g->getNodeByTensorName(                      \
                            std::vector<std::string>(outs)[0]); \
    } while (0)

namespace TNN_NS {

Status validateSetAndVector(std::set<std::string>, std::vector<std::string>);

}

#endif // TNN_SOURCE_TNN_NET_OPTIMIZER_GRAPH_MATCHER_COMMON_H_ 