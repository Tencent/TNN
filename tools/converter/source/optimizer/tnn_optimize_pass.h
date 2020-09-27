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

#ifndef TNN_TOOLS_CONVERTER_SOURCE_OPTIMIZER_TNN_OPTIMIZE_PASS_H_
#define TNN_TOOLS_CONVERTER_SOURCE_OPTIMIZER_TNN_OPTIMIZE_PASS_H_
#include "tnn/core/status.h"
#include "tnn/interpreter/net_resource.h"
#include "tnn/interpreter/net_structure.h"

namespace TNN_CONVERTER {

class TnnOptimizePass {
public:
    TnnOptimizePass()          = default;
    virtual ~TnnOptimizePass() = default;

    virtual TNN_NS::Status exec(TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource) = 0;
    virtual std::string PassName()                                                                      = 0;
};

class TnnOptimizePassManager {
public:
    TnnOptimizePassManager() = default;
    ~TnnOptimizePassManager();
    static TnnOptimizePassManager* get();
    TnnOptimizePass* search(const std::string pass_name);
    void insert(const std::string pass_name, TnnOptimizePass* t);

private:
    static TnnOptimizePassManager* tnn_optimize_pass_manager_;
    std::map<const std::string, TnnOptimizePass*> tnn_optimize_pass_map_;
};

template <class T>
class TnnOptimizePassRegister {
public:
    explicit TnnOptimizePassRegister(const std::string pass_name) {
        T* pass                                           = new T;
        TnnOptimizePassManager* tnn_optimize_pass_manager = TnnOptimizePassManager::get();
        tnn_optimize_pass_manager->insert(pass_name, pass);
    }
    ~TnnOptimizePassRegister() = default;
};
#define DECLARE_OPTIMIZE_PASS(pass_name)                                                                               \
    class TnnOptimize##pass_name##Pass : public TnnOptimizePass {                                                      \
    public:                                                                                                            \
        TnnOptimize##pass_name##Pass(){};                                                                              \
        virtual ~TnnOptimize##pass_name##Pass(){};                                                                     \
        virtual TNN_NS::Status exec(TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource);           \
        virtual std::string PassName();                                                                                \
    }

#define REGISTER_OPTIMIZE_PASS(pass_name)                                                                              \
    TnnOptimizePassRegister<TnnOptimize##pass_name##Pass> g_tnn_optimize_##pass_name##_pass_(#pass_name)
}  // namespace TNN_CONVERTER

#endif  // TNN_TOOLS_CONVERTER_SOURCE_OPTIMIZER_TNN_OPTIMIZE_PASS_H_
