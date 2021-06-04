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

#ifndef TNN_SOURCE_TNN_NETWORK_TENSORRT_LAYER_BUILDER_TENSORRT_LAYER_BUILDER_H_
#define TNN_SOURCE_TNN_NETWORK_TENSORRT_LAYER_BUILDER_TENSORRT_LAYER_BUILDER_H_

#include "tnn/network/tensorrt/layer_builder/tensorrt_base_layer_builder.h"

namespace TNN_NS {

// @brief BaseLayer Builder, defines the layer builder interface
class TensorRTLayerBuilder: public TensorRTBaseLayerBuilder {
public:
    explicit TensorRTLayerBuilder(LayerType type);

    // @brief virtual destructor
    virtual ~TensorRTLayerBuilder();

    // @brief virtual layer init
    virtual Status Init(Context* context, LayerParam* param, LayerResource* resource, std::vector<Blob*>& inputs,
                std::vector<Blob*>& outputs, AbstractDevice* device, bool enable_const_folder=true);

    // @brief layer infer
    virtual Status Forward();

    // @brief add layer to tensorRT network
    virtual ILayer* AddToNetwork(INetworkDefinition* network) = 0;

};

//@brief TensorRTTypeLayerBuilderRegister register TypeLayerBuilderCreator
template <typename T>
class TensorRTTypeLayerBuilderRegister {
public:
    explicit TensorRTTypeLayerBuilderRegister(LayerType type) {
        GetTRTLayerBuilderCreatorMap()[type] = shared_ptr<T>(new T(type));
    }
};

#define DECLARE_TENSORRT_LAYER_BUILDER(type_string, layer_type)                                                        \
    class type_string##TRTLayerBuilder : public TensorRTLayerBuilder {                                                 \
    public:                                                                                                            \
        type_string##TRTLayerBuilder(LayerType ignore) : TensorRTLayerBuilder(layer_type) {}                           \
        virtual ~type_string##TRTLayerBuilder() {}                                                                     \
        virtual ILayer* AddToNetwork(INetworkDefinition* network);                                                     \
    }

#define REGISTER_TENSORRT_LAYER_BUILDER(type_string, layer_type)                                                       \
    TensorRTTypeLayerBuilderRegister<TypeLayerBuilderCreator<type_string##TRTLayerBuilder>>                            \
        g_##layer_type##_trt_layer_builder_register(layer_type);

}  //  namespace TNN_NS

#endif  // TNN_SOURCE_TNN_NETWORK_TENSORRT_LAYER_BUILDER_TENSORRT_LAYER_BUILDER_H_
