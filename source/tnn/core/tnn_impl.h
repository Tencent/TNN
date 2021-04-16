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

#ifndef TNN_CORE_TNN_IMPL_H_
#define TNN_CORE_TNN_IMPL_H_

#include <map>
#include <memory>
#include <string>

#include "tnn/core/common.h"
#include "tnn/core/instance.h"
#include "tnn/core/status.h"

namespace TNN_NS {

class Instance;
class AbstractModelInterpreter;

class TNNImpl {
public:
    virtual ~TNNImpl();

    // @brief init the tnn, contruct model interpreter
    // @param config config model type and params
    // @return status code: Successful, returns zero. Otherwise, returns
    // error code.
    virtual Status Init(ModelConfig& config);

    // @brief release model interpreter
    virtual Status DeInit();

    //@brief Adds output to the layer. If layerName not found, then search
    // outputIndex.
    //@param output_name Name of the output blob
    //@param output_index Index of the output layer
    //@return status code: If successful, returns zero. Otherwise, returns
    // error
    // code.
    virtual Status AddOutput(const std::string& output_name, int output_index = 0) = 0;

    //@brief get input shapes map from model
    virtual Status GetModelInputShapesMap(InputShapesMap& shapes_map) = 0;

    // @brief create an instance
    // @param instance: The instance to be created.
    // @param inputs_shape: modify input shape, or it will use the shape in the
    // proto
    // @param status code: If successful, returns zero. Otherwise, returns
    // error code.
    virtual std::shared_ptr<Instance> CreateInst(NetworkConfig& config, Status& status,
                                                 InputShapesMap inputs_shape = InputShapesMap()) = 0;


    // @brief create an instance
    // @param instance: The instance to be created.
    // @param min_inputs_shape: support min shape
    // @param max_inputs_shape: support max shape
    // @param status code: If successful, returns zero. Otherwise, returns
    // error code.
    virtual std::shared_ptr<Instance> CreateInst(NetworkConfig& config, Status& status,
                                                 InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape) = 0;


protected:
    ModelConfig model_config_;
};

class AbstractTNNImplFactory {
public:
    virtual ~AbstractTNNImplFactory() {}
    virtual std::shared_ptr<TNNImpl> CreateTNNImp() = 0;
};

template <typename T>
class TNNImplFactory : public AbstractTNNImplFactory {
public:
    virtual std::shared_ptr<TNNImpl> CreateTNNImp() {
        return std::make_shared<T>();
    }
};

class TNNImplManager {
public:
    static std::shared_ptr<TNNImpl> GetTNNImpl(ModelType type);

    static void RegisterTNNImplFactory(ModelType type, AbstractTNNImplFactory* factory);

private:
    static std::map<ModelType, std::shared_ptr<AbstractTNNImplFactory>>& GetTNNImplFactoryMap();
};

template <typename T>
class TNNImplFactoryRegister {
public:
    explicit TNNImplFactoryRegister(ModelType type) {
        TNNImplManager::RegisterTNNImplFactory(type, new T());
    }
};

}  // namespace TNN_NS

#endif  // TNN_CORE_TNN_IMPL_H_
