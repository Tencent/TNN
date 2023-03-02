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

#ifndef TNN_SOURCE_TNN_OPTIMIZER_NET_OPTIMIZER_H_
#define TNN_SOURCE_TNN_OPTIMIZER_NET_OPTIMIZER_H_

#include <string>

#include "tnn/core/common.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/net_resource.h"
#include "tnn/interpreter/net_structure.h"

namespace TNN_NS {

namespace optimizer {

    //@brief net optimize must have strict rules, or error will be induced
    class NetOptimizer {
    public:
        virtual ~NetOptimizer() {}
        virtual std::string Strategy()                                          = 0;
        virtual bool IsSupported(const NetworkConfig &net_config)               = 0;
        virtual Status Optimize(NetStructure *structure, NetResource *resource) = 0;
    };

}  // namespace optimizer

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_ABSTRACT_OPTIMIZER_NET_OPTIMIZER_H_
