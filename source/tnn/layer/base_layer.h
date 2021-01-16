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

#ifndef TNN_SOURCE_TNN_LAYER_BASE_LAYER_H_
#define TNN_SOURCE_TNN_LAYER_BASE_LAYER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tnn/core/abstract_device.h"
#include "tnn/core/blob.h"
#include "tnn/core/context.h"
#include "tnn/core/layer_type.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/interpreter/layer_resource.h"

namespace TNN_NS {

//@brief BaseLaye define the layer interface
class BaseLayer {
public:
    explicit BaseLayer(LayerType type);

    // @brief virtual destructor
    virtual ~BaseLayer();

    // @brief layer init
    // @param ...
    virtual Status Init(Context* context, LayerParam* param, LayerResource* resource, std::vector<Blob*>& inputs,
                std::vector<Blob*>& outputs, AbstractDevice* device);

    //@brief Reshape recalculate the output tensor dims
    virtual Status Reshape();

    //@brief layer infer
    virtual Status Forward();

    //@brief get layer name
    std::string GetLayerName();

    //@brief set laye name
    void SetLayerName(std::string layer_name);

    //@brief get all input blobs
    virtual std::vector<Blob*> GetInputBlobs();

    //@brief get all output blobs
    virtual std::vector<Blob*> GetOutputBlobs();

    //@brief infer shape ahead for generate resource
    virtual Status InferShapeAhead(std::vector<Blob*>& input_blobs, std::vector<Blob*>& output_blobs, LayerParam* param,
                                   LayerResource* resource);
    
    // @brief set runtime bolob pool
    void SetRuntimeBlobMemoryPool(BlobMemoryPool *runtime_blob_pool);
    
    // @brief check if the layer's output is constant
    bool IsOutputConstant();
    
    // @brief set constant resource
    void SetConstantResource(ConstantResource* consts);
    
    // @brief set runtime mode
    void SetRuntimeMode(RuntimeMode mode);
protected:
    LayerType type_;

    std::string layer_name_;
    std::vector<Blob*> input_blobs_;
    std::vector<Blob*> output_blobs_;
    AbstractLayerAcc* layer_acc_;

    LayerParam* param_;
    LayerResource* resource_;
    ConstantResource* const_resource_ = nullptr;
    RuntimeMode runtime_model_ = RUNTIME_MODE_NORMAL;

    //@brief calculate the output tensor dims
    virtual Status InferOutputShape(bool ignore_error = false);
    //@brief infer the output data type, by default it is the same as input. Meanwhile, it will updata the daat flag of output blobs
    virtual Status InferOutputDataType();
    //@brief fill layer param with constant resource
    virtual Status FillLayerParamWithConstantResource();
};

//@brief LayerCreator define the create layer interface
class LayerCreator {
public:
    virtual BaseLayer* CreateLayer() = 0;
};

//@brief TypeLayerCreator create TypeLayer
template <typename T>
class TypeLayerCreator : public LayerCreator {
public:
    explicit TypeLayerCreator(LayerType type) {
        this->type_ = type;
    };
    virtual BaseLayer* CreateLayer() {
        auto layer = new T(type_);
        //        auto layer_base = dynamic_cast<BaseLayer*>(layer);
        return layer;
    }

protected:
    LayerType type_;
};

//@brief TypeLayerCreator register map
std::map<LayerType, std::shared_ptr<LayerCreator>>& GetGlobalLayerCreatorMap();

//@brief TypeLayerRegister register TypeLayerCreator
template <typename T>
class TypeLayerRegister {
public:
    explicit TypeLayerRegister(LayerType type) {
        GetGlobalLayerCreatorMap()[type] = shared_ptr<T>(new T(type));
    }
};

BaseLayer* CreateLayer(LayerType type);

#define DECLARE_LAYER_WITH_FUNC(type_string, layer_type, extra_funcs)    \
    class type_string##Layer : public BaseLayer {                                                     \
    public:                                                                                                                   \
        type_string##Layer(LayerType ignore) : BaseLayer(layer_type){};                  \
        virtual ~type_string##Layer(){};                                                                         \
                                                                                                                                  \
    protected:                                                                                                              \
        virtual Status InferOutputShape(bool ignore_error = false);                             \
        virtual Status InferOutputDataType();                                                               \
        extra_funcs \
    }

#define DECLARE_LAYER(type_string, layer_type)                                                 \
    class type_string##Layer : public BaseLayer {                                                      \
    public:                                                                                                                    \
        type_string##Layer(LayerType ignore) : BaseLayer(layer_type){};                   \
        virtual ~type_string##Layer(){};                                                                         \
                                                                                                                                   \
    protected:                                                                                                               \
        virtual Status InferOutputShape(bool ignore_error = false);                              \
        virtual Status InferOutputDataType();                                                                \
    }

#define REGISTER_LAYER(type_string, layer_type)                                                                        \
    TypeLayerRegister<TypeLayerCreator<type_string##Layer>> g_##layer_type##_register(layer_type);

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_LAYER_BASE_LAYER_H_
