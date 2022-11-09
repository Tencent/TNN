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

#ifndef TNN_TOOLS_CONVERTER_SOURCE_TORCH_TORCH_BASE_CONVERTER_H_
#define TNN_TOOLS_CONVERTER_SOURCE_TORCH_TORCH_BASE_CONVERTER_H_
#include <memory>

#include "tnn/core/common.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/net_resource.h"
#include "tnn/interpreter/net_structure.h"
#include "torch/jit.h"

namespace TNN_CONVERTER {

class TorchBaseConverter {
public:
    TorchBaseConverter()          = default;
    virtual ~TorchBaseConverter() = default;

    virtual TNN_NS::Status exec(TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource,
                                const torch::jit::Node* node, bool quantized_mode)                         = 0;
    virtual std::string TNNOpType(const torch::jit::Node* node, bool quantized_model) = 0;
    virtual TNN_NS::ActivationType ActivationType(const torch::jit::Node* node)       = 0;
    TNN_NS::Status SeparateActivation(TNN_NS::NetStructure& net_structure, TNN_NS::ActivationType activation_type);
    void InsertBlobs(TNN_NS::NetStructure& net_structure);
};

class TorchConverterManager {
public:
    TorchConverterManager() = default;
    ~TorchConverterManager();
    static TorchConverterManager* get();
    void insert(const std::string op_type, TorchBaseConverter* torch_base_converter);
    TorchBaseConverter* serach(const std::string op_type);

private:
    static TorchConverterManager* torch_converter_manager_;
    std::map<std::string, TorchBaseConverter*> torch_converter_map_;
};

template <class T>
class TorchConverterRegister {
public:
    explicit TorchConverterRegister(const std::string op_type) {
        T* converter                                   = new T;
        TorchConverterManager* torch_converter_manager = TorchConverterManager::get();
        torch_converter_manager->insert(op_type, converter);
    }
    ~TorchConverterRegister() {}
};

#define DECLARE_TORCH_OP_CONVERTER(converter_name)                                                                     \
    class Torch##converter_name##Converter : public TorchBaseConverter {                                               \
    public:                                                                                                            \
        Torch##converter_name##Converter(){};                                                                          \
        virtual ~Torch##converter_name##Converter(){};                                                                 \
        virtual TNN_NS::Status exec(TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource,            \
                                    const torch::jit::Node* node, bool quantized_mode);                                                     \
        virtual std::string TNNOpType(const torch::jit::Node* node, bool quantized_model);                             \
        virtual TNN_NS::ActivationType ActivationType(const torch::jit::Node* node);                                   \
    };

#define REGISTER_TORCH_OP_CONVERTER(converter_name, ns, op_type)                                                       \
    TorchConverterRegister<Torch##converter_name##Converter> g_torch_##ns##_##op_type##_converter_register(            \
        #ns"::"#op_type);

}  // namespace TNN_CONVERTER

#endif  // TNN_TOOLS_CONVERTER_SOURCE_TORCH_TORCH_BASE_CONVERTER_H_
