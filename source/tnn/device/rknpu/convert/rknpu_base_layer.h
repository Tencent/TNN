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

#ifndef TNN_SOURCE_TNN_RK_NPU_CONVERT_RKNPU_BASE_LAYER_ACC_H_
#define TNN_SOURCE_TNN_RK_NPU_CONVERT_RKNPU_BASE_LAYER_ACC_H_

#include <tnn/layer/base_layer.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "rknpu/rknpu_pub.h"
#include "tnn/core/abstract_device.h"
#include "tnn/core/blob.h"
#include "tnn/core/context.h"
#include "tnn/core/layer_type.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/interpreter/layer_resource.h"

namespace TNN_NS {

//@brief BaseLaye define the layer interface
class RknpuBaseLayer {
public:
    explicit RknpuBaseLayer(LayerType type);

    // @brief virtual destructor
    virtual ~RknpuBaseLayer();

    Status Init(Context *context, LayerParam *param, LayerResource *resource,
                std::vector<std::shared_ptr<rk::nn::Tensor>> input_ops, rk::nn::Graph *graph,
                std::vector<std::string> outputs);

    // @brief layer init
    // @param ...
    //@brief get layer name
    std::string GetLayerName();

    //@brief set laye name
    void SetLayerName(std::string layer_name);

    // add for rknpu
    //@brief get output operators
    std::vector<std::shared_ptr<rk::nn::Tensor>> &GetOutputOps();

    Status GetOutputShape(int i, std::vector<int> &output_shape);
    Status CalculateOutputShape(std::vector<std::vector<int>> &output_shapes);

protected:
    LayerType type_;
    std::string layer_name_;
    // add for rknpu
    rk::nn::Graph *graph_;
    std::vector<std::shared_ptr<rk::nn::Tensor>> input_ops_;
    std::vector<std::shared_ptr<rk::nn::Tensor>> output_ops_;
    LayerParam *param_;
    LayerResource *resource_;

    std::vector<std::string> outputs_name_;
    virtual Status Convert() = 0;
};

//@brief LayerCreator define the create layer interface
class RknpuLayerCreator {
public:
    virtual RknpuBaseLayer *CreateRknpuBaseLayer() = 0;
};

//@brief TypeLayerCreator create TypeLayer
template <typename T>
class TypeRknpuLayerCreator : public RknpuLayerCreator {
public:
    explicit TypeRknpuLayerCreator(LayerType type) {
        this->type_ = type;
    };
    virtual RknpuBaseLayer *CreateRknpuBaseLayer() {
        auto layer = new T(type_);
        return layer;
    }

protected:
    LayerType type_;
};

//@brief TypeLayerCreator register map
std::map<LayerType, std::shared_ptr<RknpuLayerCreator>> &GetGlobalRknpuLayerCreatorMap();

//@brief TypeLayerRegister register TypeLayerCreator
template <typename T>
class TypeRknpuLayerRegister {
public:
    explicit TypeRknpuLayerRegister(LayerType type) {
        GetGlobalRknpuLayerCreatorMap()[type] = shared_ptr<T>(new T(type));
    }
};

RknpuBaseLayer *CreateRknpuBaseLayer(LayerType type);

#define DECLARE_RKNPU_LAYER(type_string, layer_type)                                                                   \
    class Rknpu##type_string##Layer : public RknpuBaseLayer {                                                          \
    public:                                                                                                            \
        Rknpu##type_string##Layer(LayerType ignore) : RknpuBaseLayer(layer_type){};                                    \
        virtual ~Rknpu##type_string##Layer(){};                                                                        \
                                                                                                                       \
    protected:                                                                                                         \
        virtual Status Convert();                                                                                      \
    };

#define DECLARE_RKNPU_LAYER_WEIGHT(type_string, layer_type)                                                            \
    class Rknpu##type_string##Layer : public RknpuBaseLayer {                                                          \
    public:                                                                                                            \
        Rknpu##type_string##Layer(LayerType ignore) : RknpuBaseLayer(layer_type){};                                    \
        virtual ~Rknpu##type_string##Layer(){};                                                                        \
                                                                                                                       \
    protected:                                                                                                         \
        virtual Status Convert();                                                                                      \
    };

#define REGISTER_RKNPU_LAYER(type_string, layer_type)                                                                  \
    TypeRknpuLayerRegister<TypeRknpuLayerCreator<Rknpu##type_string##Layer>> g_Rknpu##layer_type##_register(layer_type);

#define ADD_OUTPUT_OP()                                                                                                \
    std::vector<std::vector<int>> output_shapes;                                                                       \
    ret = CalculateOutputShape(output_shapes);                                                                         \
    if (ret != TNN_OK)                                                                                                 \
        return ret;                                                                                                    \
    auto rk_output =                                                                                                   \
        RknpuUtils::CreateRknnTensor(graph_, outputs_name_[0], output_shapes[0], NULL, rk::nn::TensorRole::VAR);       \
    output_ops_.push_back(rk_output);

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_RK_NPU_CONVERT_RKNPU_BASE_LAYER_ACC_H_
