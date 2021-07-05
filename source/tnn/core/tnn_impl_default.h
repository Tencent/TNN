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

#ifndef TNN_CORE_TNN_IMPL_DEFAULT_H_
#define TNN_CORE_TNN_IMPL_DEFAULT_H_

#include <memory>
#include <string>

#include "tnn/core/common.h"
#include "tnn/core/instance.h"
#include "tnn/core/tnn_impl.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/abstract_model_interpreter.h"

namespace TNN_NS {

// @brief tnn impl with interpreter
class TNNImplDefault : public TNNImpl {
public:
    // @brief tnn constructor
    TNNImplDefault();

    // @brief tnn destructor
    virtual ~TNNImplDefault();

    // @brief init the tnn, construct model interpreter
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
    virtual Status AddOutput(const std::string& output_name,
                             int output_index = 0);

    //@brief get input shapes map from model
    virtual Status GetModelInputShapesMap(InputShapesMap& shapes_map);

    // @brief create an instance
    // @param instance: The instance to be created.
    // @param inputs_shape: modify input shape, or it will use the shape in the
    // proto
    // @param status code: If successful, returns zero. Otherwise, returns
    // error code.
    virtual std::shared_ptr<Instance> CreateInst(
        NetworkConfig& config, Status& status,
        InputShapesMap inputs_shape = InputShapesMap());


    // @brief create an instance
    // @param instance: The instance to be created.
    // @param min_inputs_shape: support min shape
    // @param max_inputs_shape: support max shape
    // @param status code: If successful, returns zero. Otherwise, returns
    // error code.
    virtual std::shared_ptr<Instance> CreateInst(
        NetworkConfig& config, Status& status,
        InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape);


private:
    Status AddAllLayersOutput();
    std::shared_ptr<AbstractModelInterpreter> interpreter_;
};

}  // namespace TNN_NS

#endif  // TNN_CORE_TNN_IMPL_DEFAULT_H_
