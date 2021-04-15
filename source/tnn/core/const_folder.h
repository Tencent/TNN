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

#ifndef TNN_SOURCE_TNN_CORE_CONST_FOLDER_H_
#define TNN_SOURCE_TNN_CORE_CONST_FOLDER_H_

#include <vector>

#include "tnn/core/abstract_device.h"
#include "tnn/core/abstract_network.h"
#include "tnn/core/blob.h"
#include "tnn/core/blob_manager.h"
#include "tnn/core/common.h"
#include "tnn/core/context.h"
#include "tnn/core/default_network.h"
#include "tnn/core/macro.h"
#include "tnn/core/profile.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/abstract_model_interpreter.h"
#include "tnn/interpreter/layer_resource.h"
#include "tnn/interpreter/net_resource.h"
#include "tnn/interpreter/net_structure.h"
#include "tnn/layer/base_layer.h"

namespace TNN_NS {

class ConstFolder : public DefaultNetwork {
public:
    // @brief ConstFolder Constructor
    ConstFolder();

    // @brief ConstFolder virtual Destructor
    virtual ~ConstFolder();
    
    // @brief network deinit to release init create resource
    virtual Status DeInit();
    
    // @brief get optimized NetStructure and NetResource without const layers of flag >= flag0, it must be called after Forward
    Status GetOptimizedNet(std::shared_ptr<NetStructure> &opt_structure,
                                   std::shared_ptr<NetResource> &opt_resource,
                                   int  flag0 = DATA_FLAG_CHANGE_NEVER);
    
public:
    // @brief int net with network config, net structure and net resource info
    // @param config network config info
    // @param net_structure network structure info
    // @param net_resource network resource info
    // @param inputs_shape_map modify input shape, if empty, it will use the
    // shape in proto
    virtual Status Init(NetworkConfig &net_config, ModelConfig &model_config, AbstractModelInterpreter *interpreter,
                        InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape);

    // @brief reshape with input shape info
    // @inputs input shape info
    virtual Status Reshape(const InputShapesMap &inputs);

    // @brief network forward
    virtual Status Forward();
    
protected:
    virtual Status AllocateBlobMemory();

};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_CORE_CONST_FOLDER_H_
