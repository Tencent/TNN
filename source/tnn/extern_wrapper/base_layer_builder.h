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

#ifndef TNN_SOURCE_TNN_EXTERN_WRAPPER_BASE_LAYER_BUILDER_H_
#define TNN_SOURCE_TNN_EXTERN_WRAPPER_BASE_LAYER_BUILDER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tnn/layer/base_layer.h"
#include "tnn/core/abstract_device.h"
#include "tnn/core/blob.h"
#include "tnn/core/context.h"
#include "tnn/core/layer_type.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/interpreter/layer_resource.h"
#include "tnn/extern_wrapper/foreign_tensor.h"

namespace TNN_NS {

//@brief BaseLayeer Builder, defines the layer builder interface
class BaseLayerBuilder: public BaseLayer {
public:
    explicit BaseLayerBuilder(LayerType type);

    // @brief virtual destructor
    virtual ~BaseLayerBuilder();

    // @brief layer init
    // @param ...
    Status Init(Context* context, LayerParam* param, LayerResource* resource, std::vector<Blob*>& inputs,
                std::vector<Blob*>& outputs, AbstractDevice* device);

    //@brief Reshape recalculate the output tensor dims
    virtual Status Reshape();

    //@brief layer infer
    virtual Status Forward();
protected:

    //@brief Build the foreign network 
    virtual Status Build() = 0 ;

    //@brief get all input tensors
    virtual std::vector<std::shared_ptr<ForeignTensor>> GetInputTensors();

    //@brief get all output tensors
    virtual std::vector<std::shared_ptr<ForeignTensor>> GetOutputTensors();

};

//@brief LayerBuilderCreator define the create layer interface
class LayerBuilderCreator {
public:
    virtual BaseLayerBuilder* CreateLayerBuilder() = 0;
};

//@brief TypeLayerBuilderCreator create specifiled LayerBuilder
template <typename T>
class TypeLayerBuilderCreator : public LayerBuilderCreator {
public:
    explicit TypeLayerBuilderCreator(LayerType type) {
        this->type_ = type;
    };
    virtual BaseLayerBuilder* CreateLayerBuilder() {
        auto layer_builder = new T(type_);
        return layer_builder;
    }

protected:
    LayerType type_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_EXTERN_WRAPPER_BASE_LAYER_BUILDER_H_
