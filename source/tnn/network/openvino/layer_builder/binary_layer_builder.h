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

#ifndef TNN_SOURCE_TNN_DEVICE_OPENVINO_LAYER_BUILDER_BINARY_LAYER_BUILDER_H_
#define TNN_SOURCE_TNN_DEVICE_OPENVINO_LAYER_BUILDER_BINARY_LAYER_BUILDER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <ngraph/node.hpp>
#include <ngraph/op/op.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "tnn/interpreter/layer_param.h"
#include "tnn/interpreter/layer_resource.h"
#include "tnn/extern_wrapper/base_layer_builder.h"
#include "tnn/network/openvino/openvino_types.h"
#include "tnn/network/openvino/layer_builder/openvino_layer_builder.h"

namespace TNN_NS {
namespace openvino {

//@brief BaseLayeer Builder, defines the layer builder interface
class BinaryLayerBuilder: public OpenVINOLayerBuilder {
public:
    // @brief virtual destructor
    explicit BinaryLayerBuilder(LayerType type): OpenVINOLayerBuilder(type) {};

    // @brief virtual destructor
    virtual ~BinaryLayerBuilder() {};

protected:
    virtual Status InferOutputShape() {return TNN_OK;};
    virtual Status InferOutputDataType() {return TNN_OK;};
    virtual Status Build();            
    virtual std::shared_ptr<ngraph::Node> CreateNode(const ngraph::Output<ngraph::Node>& arg0, 
                                                     const ngraph::Output<ngraph::Node>& arg1) = 0;
};

#define DECLARE_BINARY_LAYER_BUILDER(type_string, layer_type)                                                          \
    class type_string##LayerBuilder : public BinaryLayerBuilder {                                                      \
    public:                                                                                                            \
        using ngraph_node_type = ngraph::op::v1::type_string;                                                          \
        type_string##LayerBuilder(LayerType ignore) : BinaryLayerBuilder(layer_type){};                                \
        virtual ~type_string##LayerBuilder(){};                                                                        \
                                                                                                                       \
    protected:                                                                                                         \
        virtual std::shared_ptr<ngraph::Node> CreateNode(const ngraph::Output<ngraph::Node>& arg0,                     \
                                                         const ngraph::Output<ngraph::Node>& arg1) {                   \
            return std::make_shared<ngraph_node_type>(arg0, arg1);                                                     \
        }                                                                                                              \
    };

#define REGISTER_BINARY_LAYER_BUILDER(type_string, layer_type)                                                       \
    TypeLayerBuilderRegister<TypeLayerBuilderCreator<type_string##LayerBuilder>>                                     \
        g_##layer_type##_layer_builder_register(layer_type);

}  // namespace openvino
}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_OPENVINO_LAYER_BUILDER_BINARY_LAYER_BUILDER_H_
