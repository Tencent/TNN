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

#ifndef TNN_SOURCE_TNN_INTERPRETER_LAYER_RESOURCE_GENERATOR_H_
#define TNN_SOURCE_TNN_INTERPRETER_LAYER_RESOURCE_GENERATOR_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tnn/core/blob.h"
#include "tnn/core/layer_type.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/interpreter/layer_resource.h"

namespace TNN_NS {
//@brief random gen layer resource in benchmark mode, save upload model time
class LayerResourceGenerator {
public:
    virtual Status GenLayerResource(LayerParam* param, LayerResource** resource, std::vector<Blob*>& inputs) = 0;
    virtual Status ConvertHalfLayerResource(LayerResource* src_res, LayerResource** dst_res)                 = 0;
};

std::map<LayerType, std::shared_ptr<LayerResourceGenerator>>& GetGlobalLayerResourceGeneratorMap();

Status GenerateRandomResource(LayerType type, LayerParam* param, LayerResource** resource, std::vector<Blob*>& inputs);

//@brief only convert iterms of half data type to fp32 data type
Status ConvertHalfResource(LayerType type, LayerResource* src_res, LayerResource** dst_res);

template <typename T>
class TypeLayerResourceRegister {
public:
    explicit TypeLayerResourceRegister(LayerType type) {
        GetGlobalLayerResourceGeneratorMap()[type] = shared_ptr<T>(new T);
    }
};

#define REGISTER_LAYER_RESOURCE(type_string, layer_type)                                                               \
    TypeLayerResourceRegister<type_string##LayerResourceGenerator> g_##layer_type##_resource_register(layer_type);

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_INTERPRETER_LAYER_RESOURCE_GENERATOR_H_
