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

#ifndef TNN_TOOLS_CONVERTER_SOURCE_ONNX_ONNX_BASE_CONVERTER_H_
#define TNN_TOOLS_CONVERTER_SOURCE_ONNX_ONNX_BASE_CONVERTER_H_
#include <memory>

#include "onnx.pb.h"
#include "onnx_proxy_graph.h"
#include "tnn/interpreter/net_resource.h"
#include "tnn/interpreter/net_structure.h"

namespace TNN_CONVERTER {

class OnnxBaseConverter {
public:
    OnnxBaseConverter()          = default;
    virtual ~OnnxBaseConverter() = default;

    virtual TNN_NS::Status exec(TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource,
                                const onnx::NodeProto& node,
                                std::map<std::string, const onnx::TensorProto*>& proxy_initializers_map,
                                std::map<std::string, std::shared_ptr<OnnxProxyNode>>& proxy_nodes,
                                bool& quantized_model)                               = 0;
    virtual std::string TNNOpType(const onnx::NodeProto& node, bool quantized_model) = 0;
    virtual TNN_NS::ActivationType ActivationType(const onnx::NodeProto& node)       = 0;
    TNN_NS::Status SeparateActivation(TNN_NS::NetStructure& net_structure, TNN_NS::ActivationType activation_type);
    void InsertBlobs(TNN_NS::NetStructure& net_structure);

protected:
    const onnx::NodeProto* FindNodeProto(const std::string& name,
                                         std::map<std::string, std::shared_ptr<OnnxProxyNode>> proxy_nodes);
};

class OnnxConverterManager {
public:
    OnnxConverterManager() = default;
    ~OnnxConverterManager();
    static OnnxConverterManager* get();
    void insert(const std::string onnx_op_type, OnnxBaseConverter* onnx_base_converter);
    OnnxBaseConverter* search(const std::string onnx_op_type);

private:
    static OnnxConverterManager* onnx_converter_manager_;
    std::map<std::string, OnnxBaseConverter*> onnx_converter_map_;
};
template <class T>
class OnnxConverterRegister {
public:
    explicit OnnxConverterRegister(const std::string onnx_op_type) {
        T* converter                                 = new T;
        OnnxConverterManager* onnx_converter_manager = OnnxConverterManager::get();
        onnx_converter_manager->insert(onnx_op_type, converter);
    }
    ~OnnxConverterRegister(){};
};

#define DECLARE_OP_CONVERTER(op_converter_name)                                                                        \
    class Onnx##op_converter_name##Converter : public OnnxBaseConverter {                                              \
    public:                                                                                                            \
        Onnx##op_converter_name##Converter(){};                                                                        \
        virtual ~Onnx##op_converter_name##Converter(){};                                                               \
        virtual TNN_NS::Status exec(TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource,            \
                                    const onnx::NodeProto& node,                                                       \
                                    std::map<std::string, const onnx::TensorProto*>& proxy_initializers_map,           \
                                    std::map<std::string, std::shared_ptr<OnnxProxyNode>>& proxy_nodes,                \
                                    bool& quantized_model);                                                            \
        virtual std::string TNNOpType(const onnx::NodeProto& node, bool quantized_model);                              \
        virtual TNN_NS::ActivationType ActivationType(const onnx::NodeProto& node);                                    \
    }

#define REGISTER_CONVERTER(op_converter_name, onnx_op_type)                                                            \
    OnnxConverterRegister<Onnx##op_converter_name##Converter> g_converter_##onnx_op_type##_(#onnx_op_type)

}  // namespace TNN_CONVERTER

#endif  // TNN_TOOLS_CONVERTER_SOURCE_ONNX_ONNX_BASE_CONVERTER_H_
