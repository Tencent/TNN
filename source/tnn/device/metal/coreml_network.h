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

#ifndef TNN_SOURCE_TNN_DEVICE_METAL_CORE_COREML_NETWORK_H_
#define TNN_SOURCE_TNN_DEVICE_METAL_CORE_COREML_NETWORK_H_
#import <CoreML/CoreML.h>

#include "tnn/core/abstract_device.h"
#include "tnn/core/abstract_network.h"
#include "tnn/core/context.h"

namespace TNN_NS {

class CoreMLNetwork : public AbstractNetwork {
public:
    // @brief CoreMLNetwork Constructor
    CoreMLNetwork();

    // @brief CoreMLNetwork virtual Destructor
    virtual ~CoreMLNetwork();

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

    // @brief get tnn command queue
    // @param command_queue device command queue for forward
    virtual Status GetCommandQueue(void **command_queue);

    // @brief network forward
    virtual Status Forward();

    // @brief tnn instance network infer, it will not wait
    virtual Status ForwardAsync(Callback call_back);

    // @brief network deinit to release init create resource
    virtual Status DeInit();

    // @brief get network forward for all blob memory size
    virtual Status GetForwardMemorySize(int &memory_size);

    // @brief set forward memory when share memory mode is set from external
    virtual Status SetForwardMemory(void *memory);

    // @brief get all input blobs
    virtual Status GetAllInputBlobs(BlobMap &blobs);

    // @brief get all output blobs
    virtual Status GetAllOutputBlobs(BlobMap &blobs);

private:
    AbstractDevice *device_              = nullptr;
    Context *context_                    = nullptr;
    __strong NSDictionary *coreml_net_   = nil;
    __strong NSDictionary *coreml_shape_ = nil;
    __strong NSObject *coreml_model_     = nil;
    BlobMap blob_input_map_;
    BlobMap blob_output_map_;
    DimsVector coreml_input_dims_;
    DimsVector coreml_output_dims_;
    
    Status CheckCoreMLStatus();
};

} // namespace TNN_NS

#endif // TNN_SOURCE_TNN_DEVICE_METAL_CORE_COREML_NETWORK_H_
