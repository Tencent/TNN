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

#ifndef TNN_INCLUDE_TNN_CORE_TNN_H_
#define TNN_INCLUDE_TNN_CORE_TNN_H_

#include <memory>
#include <string>

#include "tnn/core/blob.h"
#include "tnn/core/common.h"
#include "tnn/core/instance.h"
#include "tnn/core/macro.h"
#include "tnn/core/status.h"

#pragma warning(push)
#pragma warning(disable:4251)

namespace TNN_NS {

class TNNImpl;

class PUBLIC TNN {
public:
    TNN();
    
    ~TNN();

    // init tnn implement, interpret model.
    Status Init(ModelConfig& config);

    // denit tnn implement, release model interpreter.
    Status DeInit();

    // add output to the model. 
    // if output_name of blob not found, then search output_index of layer.
    Status AddOutput(const std::string& output_name, int output_index = 0);

    // return input shapes map from model
    Status GetModelInputShapesMap(InputShapesMap& shapes_map);

    // create tnn network instance with network config and inputs shape.
    // if inputs shape not set, use default from model.
    std::shared_ptr<Instance> CreateInst(
        NetworkConfig& config, Status& status,
        InputShapesMap inputs_shape = InputShapesMap());

    // create tnn network instance with network config and min max inputs shape,
    // instance reshape can support range from min inputs shape to max inputs shape.
    std::shared_ptr<Instance> CreateInst(
        NetworkConfig& config, Status& status,
        InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape);

private:
    std::shared_ptr<TNNImpl> impl_ = nullptr;
};

}  // namespace TNN_NS

#pragma warning(pop)

#endif  // TNN_INCLUDE_TNN_CORE_TNN_H_
