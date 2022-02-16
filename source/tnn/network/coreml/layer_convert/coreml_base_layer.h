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

#ifndef TNN_SOURCE_TNN_DEVICE_COREML_LAYER_CONVERT_COREML_BASE_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_COREML_LAYER_CONVERT_COREML_BASE_LAYER_ACC_H_

#include <tnn/layer/base_layer.h>
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
#include "tnn/interpreter/default_model_interpreter.h"
#include "../mlmodel/include/Model.pb-c.h"

#ifndef TNN_COREML_FULL_PRECISION
#define TNN_COREML_FULL_PRECISION 1
#endif

namespace TNN_NS {

std::shared_ptr<char> NullTerminatedCString(std::string & name);

class CoreMLConstLayer;

//@brief BaseLaye define the layer interface
class CoreMLBaseLayer {
public:
    explicit CoreMLBaseLayer(LayerType type);

    // @brief virtual destructor
    virtual ~CoreMLBaseLayer();
    Status Init(LayerInfo* layer_info ,LayerResource *layer_resource);
    void SetNetResource(NetResource *net_resource);
    
    // @brief layer init
    // @param ...
    //@brief get layer name
    virtual std::string GetLayerName();
    
    // @brief get internal coreml layers, include const weight input layer
    virtual std::vector<CoreML__Specification__NeuralNetworkLayer*> GetCoreMLLayerPtrs();
    
    // @brief convert to coreml layer
    Status Convert();
protected:
    // @brief set coreml layer type
    virtual Status BuildLayerType();
    // @brief set coreml layer param
    virtual Status BuildLayerParam();
    // @brief convert weights to coreml const layer
    virtual Status BuildConstantWeightsLayer();
    /* @brief generate all inputs of coreml layer
     * For TNN op without input from layresource, you dont need override this func, it will generate all inputs form layer info automatically;
     * For TNN op with input from layresource, you must override this func to generate all inputs manually;
     */
    virtual std::vector<std::string> BuildLayerInputs();
    /* @brief generate all outputs of coreml layer
     * For TNN op without output from layresource, you dont need override this func, it will generate all outputs form layer info automatically;
     * For TNN op with output from layresource, you must override this func to generate all outputs manually;
     */
    virtual std::vector<std::string> BuildLayerOutputs();

    //@brief set layer name
    void SetLayerName(std::string& name);
    
    //@brief set outputs for coreml layer
    void SetLayerInputs(std::vector<std::string>& inputs);
    //@brief set outputs for coreml layer
    void SetLayerOutputs(std::vector<std::string>& outputs);
    
protected:
    LayerType type_;
    LayerInfo* layer_info_ = nullptr;
    LayerResource *layer_resource_  = nullptr;
    NetResource *net_resource_ = nullptr;
    
    std::unique_ptr<CoreML__Specification__NeuralNetworkLayer> coreml_layer_;
    std::shared_ptr<char> coreml_layer_name_;
    std::shared_ptr<void> coreml_layer_param_;
    std::shared_ptr<char*> coreml_layer_inputs_arr_;
    std::vector<std::shared_ptr<char> > coreml_layer_inputs_;
    std::shared_ptr<char*> coreml_layer_outputs_arr_;
    std::vector<std::shared_ptr<char> > coreml_layer_outputs_;

    //for some op such as add, conv, weight value is stored in layer resource, so constant coreml layer is needed for every layer
    std::vector<std::shared_ptr<CoreMLConstLayer> > coreml_layer_constant_weights_;
    //for some op to add op before itself
    std::shared_ptr<CoreMLBaseLayer> coreml_layer_before_;
    //for some op to add op after itself
    std::shared_ptr<CoreMLBaseLayer> coreml_layer_after_;
    
};

//@brief LayerCreator define the create layer interface
class CoreMLLayerCreator {
public:
    virtual std::shared_ptr<CoreMLBaseLayer> CreateCoreMLBaseLayer() = 0;
};

//@brief TypeLayerCreator create TypeLayer
template <typename T>
class TypeCoreMLLayerCreator : public CoreMLLayerCreator {
public:
    explicit TypeCoreMLLayerCreator(LayerType type) {
        this->type_ = type;
    };
    virtual std::shared_ptr<CoreMLBaseLayer> CreateCoreMLBaseLayer() {
        auto layer = std::shared_ptr<CoreMLBaseLayer>(new T(type_));
        return layer;
    }

protected:
    LayerType type_;
};

//@brief TypeLayerCreator register map
std::map<LayerType, std::shared_ptr<CoreMLLayerCreator>> &GetGlobalCoreMLLayerCreatorMap();

//@brief TypeLayerRegister register TypeLayerCreator
template <typename T>
class TypeCoreMLLayerRegister {
public:
    explicit TypeCoreMLLayerRegister(LayerType type) {
        GetGlobalCoreMLLayerCreatorMap()[type] = shared_ptr<T>(new T(type));
    }
};

std::shared_ptr<CoreMLBaseLayer> CreateCoreMLBaseLayer(LayerType type);

#define DECLARE_COREML_BINARY_LAYER(type_string, layer_type)                                \
  class CoreML##type_string##Layer : public CoreMLBinaryLayer {                             \
    public:                                                                                 \
        CoreML##type_string##Layer(LayerType layer_type) : CoreMLBinaryLayer(layer_type){}; \
        virtual ~CoreML##type_string##Layer(){};                                            \
    protected:                                                                              \
        virtual Status BuildLayerType();                                                    \
        virtual Status BuildLayerParam();                                                   \
        virtual Status BuildConstantWeightsLayer();                                         \
        virtual std::vector<std::string> BuildLayerInputs();                                \
        virtual std::vector<std::string> BuildLayerOutputs();                               \
    }

#define DECLARE_COREML_LAYER(type_string, layer_type)                                       \
  class CoreML##type_string##Layer : public CoreMLBaseLayer {                               \
    public:                                                                                 \
        CoreML##type_string##Layer(LayerType layer_type) : CoreMLBaseLayer(layer_type){};   \
        virtual ~CoreML##type_string##Layer(){};                                            \
    protected:                                                                              \
        virtual Status BuildLayerType();                                                    \
        virtual Status BuildLayerParam();                                                   \
        virtual Status BuildConstantWeightsLayer();                                         \
        virtual std::vector<std::string> BuildLayerInputs();                                \
        virtual std::vector<std::string> BuildLayerOutputs();                               \
    }

#define DECLARE_COREML_LAYER_WITH_DATA(type_string, layer_type, extra_datas)                \
  class CoreML##type_string##Layer : public CoreMLBaseLayer {                               \
    public:                                                                                 \
        CoreML##type_string##Layer(LayerType layer_type) : CoreMLBaseLayer(layer_type){};   \
        virtual ~CoreML##type_string##Layer(){};                                            \
    protected:                                                                              \
        virtual Status BuildLayerType();                                                    \
        virtual Status BuildLayerParam();                                                   \
        virtual Status BuildConstantWeightsLayer();                                         \
        virtual std::vector<std::string> BuildLayerInputs();                                \
        virtual std::vector<std::string> BuildLayerOutputs();                               \
        extra_datas                                                                         \
    }

#define DECLARE_COREML_LAYER_WITH_FUNC_DATA(type_string, layer_type, extra_funcs, extra_datas)       \
  class CoreML##type_string##Layer : public CoreMLBaseLayer {                                        \
    public:                                                                                          \
        CoreML##type_string##Layer(LayerType layer_type) : CoreMLBaseLayer(layer_type){};            \
        virtual ~CoreML##type_string##Layer(){};                                                     \
        extra_funcs                                                                                  \
    protected:                                                                                       \
        virtual Status BuildLayerType();                                                             \
        virtual Status BuildLayerParam();                                                            \
        virtual Status BuildConstantWeightsLayer();                                                  \
        virtual std::vector<std::string> BuildLayerInputs();                                         \
        virtual std::vector<std::string> BuildLayerOutputs();                                        \
        extra_datas                                                                                  \
    }

#define REGISTER_COREML_LAYER(type_string, layer_type)                                                                       \
    TypeCoreMLLayerRegister<TypeCoreMLLayerCreator<CoreML##type_string##Layer>> g_CoreML##layer_type##_register(layer_type);

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_COREML_LAYER_CONVERT_COREML_BASE_LAYER_ACC_H_
