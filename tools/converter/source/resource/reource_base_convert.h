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

#ifndef TNN_TOOLS_CONVERTER_SOURCE_RESOURCE_REOURCE_BASE_CONVERT_H_
#define TNN_TOOLS_CONVERTER_SOURCE_RESOURCE_REOURCE_BASE_CONVERT_H_
#include <map>

#include "tnn/core/common.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/interpreter/net_resource.h"
#include "tnn/interpreter/net_structure.h"

namespace TNN_CONVERTER {
class ResourceBaseConvert {
public:
    ResourceBaseConvert()                                                                               = default;
    virtual ~ResourceBaseConvert()                                                                      = default;
    virtual TNN_NS::Status ConvertToHalfResource(std::shared_ptr<TNN_NS::LayerParam> param,
                                                 std::shared_ptr<TNN_NS::LayerResource> layer_resource) = 0;
};

class ResourceConvertManager {
public:
    ResourceConvertManager() = default;
    ~ResourceConvertManager();
    static ResourceConvertManager* get();
    void insert(const std::string& tnn_op_name, ResourceBaseConvert* resource_base_convert);
    ResourceBaseConvert* search(const std::string& tnn_op_name);

private:
    static ResourceConvertManager* resource_convert_manager_;
    std::map<std::string, ResourceBaseConvert*> resource_convert_map_;
};
template <class T>
class ResourceConvertRegister {
public:
    explicit ResourceConvertRegister(const std::string& tnn_op_name) {
        T* convert                                       = new T;
        ResourceConvertManager* resource_convert_manager = ResourceConvertManager::get();
        resource_convert_manager->insert(tnn_op_name, convert);
    }
};

#define REGISTER_RESOURCE_CONVERT(op_convert_name, tnn_op_name)                                                        \
    ResourceConvertRegister<Resource##op_convert_name##Convert> g_resource_converter_##tnn_op_name##_(#tnn_op_name)

}  // namespace TNN_CONVERTER

#define DECLARE_RESOURCE_CONVERT(op_convert_name)                                                                      \
    class Resource##op_convert_name##Convert : public ResourceBaseConvert {                                            \
    public:                                                                                                            \
        Resource##op_convert_name##Convert(){};                                                                        \
        virtual ~Resource##op_convert_name##Convert(){};                                                               \
        virtual TNN_NS::Status ConvertToHalfResource(std::shared_ptr<TNN_NS::LayerParam> param,                        \
                                                     std::shared_ptr<TNN_NS::LayerResource> layer_resource);           \
    };

#endif  // TNN_TOOLS_CONVERTER_SOURCE_RESOURCE_REOURCE_BASE_CONVERT_H_
