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

#ifndef TNN_SOURCE_TNN_DEVICE_OPENVINO_LAYER_BUILDER_OPENVINO_LAYER_BUILDER_H_
#define TNN_SOURCE_TNN_DEVICE_OPENVINO_LAYER_BUILDER_OPENVINO_LAYER_BUILDER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <ngraph/node.hpp>
#include <ngraph/op/op.hpp>

#include "tnn/layer/base_layer.h"
#include "tnn/core/abstract_device.h"
#include "tnn/core/blob.h"
#include "tnn/core/context.h"
#include "tnn/core/layer_type.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/interpreter/layer_resource.h"
#include "tnn/extern_wrapper/base_layer_builder.h"

namespace TNN_NS {

//@brief BaseLayeer Builder, defines the layer builder interface
class OpenVINOLayerBuilder: public BaseLayerBuilder {
public:
    explicit OpenVINOLayerBuilder(LayerType type);

    // @brief virtual destructor
    virtual ~OpenVINOLayerBuilder();

    // @brief layer init
    // @param ...
    Status Init(Context* context, LayerParam* param, LayerResource* resource, std::vector<Blob*>& inputs,
                std::vector<Blob*>& outputs, AbstractDevice* device);

    //@brief Reshape recalculate the output tensor dims
    virtual Status Reshape();

    //@brief layer infer
    virtual Status Forward();

    ngraph::element::Type_t DataTransfer(DataType type) {
        return dataTypeTransfer[type];
    }
protected:

    //@brief get all input nodes
    virtual std::vector<std::shared_ptr<ngraph::Node>> GetInputNodes();

    //@brief get all input nodes
    virtual std::vector<std::shared_ptr<ngraph::Node>> GetOutputNodes();

    virtual Status SetOutputNodes(ngraph::NodeVector);

    virtual Status SetOutputTensors(std::vector<Blob*>);
    //@brief Build the foreign network 
    virtual Status Build() = 0 ;

    virtual LayerResource* GetResource();

    std::vector<std::shared_ptr<ngraph::Node>> inputNodes_;
    std::vector<std::shared_ptr<ngraph::Node>> outputNodes_;

    std::map<DataType, ngraph::element::Type_t> dataTypeTransfer = {
        {DATA_TYPE_FLOAT, ngraph::element::Type_t::f32},
        {DATA_TYPE_BFP16, ngraph::element::Type_t::bf16},
        {DATA_TYPE_HALF, ngraph::element::Type_t::f16},
        {DATA_TYPE_INT32, ngraph::element::Type_t::i32},
        {DATA_TYPE_INT8, ngraph::element::Type_t::i8}
    };
};

//@brief TypeLayerBuilderCreator register map
std::map<LayerType, std::shared_ptr<LayerBuilderCreator>>& GetOpenVINOLayerBuilderCreatorMap();

//@brief TypeLayerBuilderRegister register TypeLayerBuilderCreator
template <typename T>
class TypeLayerBuilderRegister {
public:
    explicit TypeLayerBuilderRegister(LayerType type) {
        GetOpenVINOLayerBuilderCreatorMap()[type] = shared_ptr<T>(new T(type));
    }
};

OpenVINOLayerBuilder* CreateOpenVINOLayerBuilder(LayerType type);

#define DECLARE_OPENVINO_LAYER_BUILDER(type_string, layer_type)                                                        \
    class type_string##OVLayerBuilder : public OpenVINOLayerBuilder {                                                  \
    public:                                                                                                            \
        type_string##OVLayerBuilder(LayerType ignore) : OpenVINOLayerBuilder(layer_type){};                            \
        virtual ~type_string##OVLayerBuilder(){};                                                                      \
                                                                                                                       \
    protected:                                                                                                         \
        virtual Status InferOutputShape() {return TNN_OK;};                                                            \
        virtual Status InferOutputDataType() {return TNN_OK;};                                                         \
        virtual Status Build();                                                                                        \
    }

#define REGISTER_OPENVINO_LAYER_BUILDER(type_string, layer_type)                                                       \
    TypeLayerBuilderRegister<TypeLayerBuilderCreator<type_string##OVLayerBuilder>>                                     \
        g_##layer_type##_ov_layer_builder_register(layer_type);

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_OPENVINO_LAYER_BUILDER_OPENVINO_LAYER_BUILDER_H_
