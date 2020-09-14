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

#ifndef TNN_SOURCE_TNN_NPU_CONVERT_NPU_BASE_LAYER_CONVERT_ACC_H_
#define TNN_SOURCE_TNN_NPU_CONVERT_NPU_BASE_LAYER_CONVERT_ACC_H_

#include <tnn/layer/base_layer.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "graph/attr_value.h"
#include "graph/op/nn_defs.h"
#include "graph/compatible/all_ops.h"
#include "tnn/core/abstract_device.h"
#include "tnn/core/blob.h"
#include "tnn/core/context.h"
#include "tnn/core/layer_type.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/interpreter/layer_resource.h"

namespace TNN_NS {

class OperatorInfo {
public:
    OperatorInfo();
    explicit OperatorInfo(std::shared_ptr<ge::Operator> op);
    OperatorInfo(std::shared_ptr<ge::Operator> op, vector<int> shape);

    virtual ~OperatorInfo();

    shared_ptr<ge::Operator> GetOperator();
    std::vector<int> GetShape();
    void SetShape(vector<int> shape);
    void SetOperator(std::shared_ptr<ge::Operator> op);

private:
    std::shared_ptr<ge::Operator> op_;
    std::vector<int> shape_;
};

//@brief BaseLaye define the layer interface
class NpuBaseLayer {
public:
    explicit NpuBaseLayer(LayerType type);

    // @brief virtual destructor
    virtual ~NpuBaseLayer();

    Status Init(Context *context, LayerParam *param, LayerResource *resource,
                std::vector<std::shared_ptr<OperatorInfo>> input_ops, AbstractDevice *device,
                std::vector<std::string> outputs);

    // @brief layer init
    // @param ...
    //@brief get layer name
    std::string GetLayerName();

    //@brief set laye name
    void SetLayerName(std::string layer_name);

    // add for huawei_npu
    //@brief get output operators
    std::vector<std::shared_ptr<OperatorInfo>> &GetOutputOps();

    Status SetOutputOps();
    Status GetOutputShape(int i, std::vector<int> &output_shape);
    Status CalculateOutputShape(std::vector<std::vector<int>> &output_shapes);

protected:
    LayerType type_;
    std::string layer_name_;
    // add for huawei_npu
    std::vector<std::shared_ptr<OperatorInfo>> input_ops_;
    std::vector<std::shared_ptr<OperatorInfo>> output_ops_;
    LayerParam *param_;
    LayerResource *resource_;

    std::vector<std::string> outputs_name_;
    virtual Status Convert() = 0;
};

//@brief LayerCreator define the create layer interface
class NpuLayerCreator {
public:
    virtual NpuBaseLayer *CreateNpuBaseLayer() = 0;
};

//@brief TypeLayerCreator create TypeLayer
template <typename T>
class TypeNpuLayerCreator : public NpuLayerCreator {
public:
    explicit TypeNpuLayerCreator(LayerType type) {
        this->type_ = type;
    };
    virtual NpuBaseLayer *CreateNpuBaseLayer() {
        auto layer = new T(type_);
        return layer;
    }

protected:
    LayerType type_;
};

//@brief TypeLayerCreator register map
std::map<LayerType, std::shared_ptr<NpuLayerCreator>> &GetGlobalNpuLayerCreatorMap();

//@brief TypeLayerRegister register TypeLayerCreator
template <typename T>
class TypeNpuLayerRegister {
public:
    explicit TypeNpuLayerRegister(LayerType type) {
        GetGlobalNpuLayerCreatorMap()[type] = shared_ptr<T>(new T(type));
    }
};

NpuBaseLayer *CreateNpuBaseLayer(LayerType type);

#define DECLARE_NPU_LAYER(type_string, layer_type)                                                                     \
    class Npu##type_string##Layer : public NpuBaseLayer {                                                              \
    public:                                                                                                            \
        Npu##type_string##Layer(LayerType ignore) : NpuBaseLayer(layer_type){};                                        \
        virtual ~Npu##type_string##Layer(){};                                                                          \
                                                                                                                       \
    protected:                                                                                                         \
        virtual Status Convert();                                                                                      \
    };

#define DECLARE_NPU_LAYER_WEIGHT(type_string, layer_type)                                                              \
    class Npu##type_string##Layer : public NpuBaseLayer {                                                              \
    public:                                                                                                            \
        Npu##type_string##Layer(LayerType ignore) : NpuBaseLayer(layer_type){};                                        \
        virtual ~Npu##type_string##Layer(){};                                                                          \
                                                                                                                       \
    protected:                                                                                                         \
        virtual Status Convert();                                                                                      \
        std::vector<std::shared_ptr<ge::Operator>> weight_ops_;                                                        \
    };

#define REGISTER_NPU_LAYER(type_string, layer_type)                                                                    \
    TypeNpuLayerRegister<TypeNpuLayerCreator<Npu##type_string##Layer>> g_Npu##layer_type##_register(layer_type);

#define ADD_OUTPUT_OP(output)                                                                                          \
    std::shared_ptr<OperatorInfo> output_op = std::make_shared<OperatorInfo>(output);                                  \
    output_ops_.push_back(output_op);                                                                                  \
    return SetOutputOps();

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_NPU_CONVERT_NPU_BASE_LAYER_CONVERT_ACC_H_
