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

#ifndef TNN_SOURCE_TNN_INTERPRETER_NCNN_OPTIMIZER_NCNN_OPTMIZER_H_
#define TNN_SOURCE_TNN_INTERPRETER_NCNN_OPTIMIZER_NCNN_OPTMIZER_H_

#include <string>

#include "tnn/core/common.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/net_resource.h"
#include "tnn/interpreter/net_structure.h"

namespace TNN_NS {

namespace ncnn {

    //@brief interprete ncnn models to tnn models
    class NCNNOptimizer {
    public:
        NCNNOptimizer(){};
        virtual std::string Strategy() = 0;
        virtual ~NCNNOptimizer(){};
        virtual Status Optimize(NetStructure *structure, NetResource *resource) = 0;
    };

#define DECLARE_NCNN_OPTIMIZER(optimizer_name)                                                                         \
    class optimizer_name##Optimizer : public NCNNOptimizer {                                                           \
    public:                                                                                                            \
        virtual std::string Strategy();                                                                                \
        virtual Status Optimize(NetStructure *structure, NetResource *resource);                                       \
    }

}  // namespace ncnn

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_INTERPRETER_NCNN_OPTIMIZER_NCNN_OPTMIZER_H_
