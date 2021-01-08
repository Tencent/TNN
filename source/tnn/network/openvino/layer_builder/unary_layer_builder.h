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

#ifndef TNN_SOURCE_TNN_DEVICE_OPENVINO_LAYER_UNARY_BINARY_LAYER_BUILDER_H_
#define TNN_SOURCE_TNN_DEVICE_OPENVINO_LAYER_UNARY_BINARY_LAYER_BUILDER_H_

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
template<class NGRAPH_OP_TYPE>
class UnaryLayerBuilder: public OpenVINOLayerBuilder {
public:
    // @brief virtual destructor
    explicit UnaryLayerBuilder(LayerType type): OpenVINOLayerBuilder(type) {};

    // @brief virtual destructor
    virtual ~UnaryLayerBuilder() {};

protected:
    virtual Status InferOutputShape() {return TNN_OK;};
    virtual Status InferOutputDataType() {return TNN_OK;};
    virtual Status Build() {

        if (GetInputNodes().size() <=0) {
            LOGE("Error: 0 input nodes\n");
            return TNNERR_INIT_LAYER;
        }
        auto input_node = GetInputNodes()[0];
        auto unary_node = std::make_shared<NGRAPH_OP_TYPE>(input_node->output(0));

        unary_node->set_friendly_name(param_->name);
        unary_node->validate_and_infer_types();

        SetOutputTensors(ngraph::NodeVector({unary_node}));

        return TNN_OK;    
    }
};

#define DECLARE_UNARY_LAYER_BUILDER(type_string, layer_type)                                                           \
    class type_string##LayerBuilder : public UnaryLayerBuilder<ngraph::op::type_string> {                              \
    public:                                                                                                            \
        type_string##LayerBuilder(LayerType ignore) : UnaryLayerBuilder(layer_type){};                                 \
        virtual ~type_string##LayerBuilder(){};                                                                        \
    };

#define REGISTER_UNARY_LAYER_BUILDER(type_string, layer_type)                                                          \
    TypeLayerBuilderRegister<TypeLayerBuilderCreator<type_string##LayerBuilder>>                                       \
        g_##layer_type##_layer_builder_register(layer_type);

}  // namespace openvino
}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_OPENVINO_LAYER_UNARY_BINARY_LAYER_BUILDER_H_O
