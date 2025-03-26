// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef TNN_SOURCE_TNN_DEVICE_SNPE_SNPE_NETWORK_H_
#define TNN_SOURCE_TNN_DEVICE_SNPE_SNPE_NETWORK_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "SNPE/SNPE.hpp"

#include "tnn/core/abstract_network.h"
#include "tnn/core/default_network.h"

namespace TNN_NS {

class SnpeNetwork : public DefaultNetwork {
public:
    // @brief virtual default destructor
    virtual ~SnpeNetwork();

    // @brief int net with network config, net structure and net resource info
    // @param config network config info
    // @param net_structure network structure info
    // @param net_resource network resource info
    // @param inputs_shape_map modify input shape, if empty, it will use the shape in proto
    // @param inputs_data_type specify input data type, by default float
    virtual Status Init(NetworkConfig &net_config,
                        ModelConfig &model_config,
                        AbstractModelInterpreter *interpreter,
                        InputShapesMap min_inputs_shape,
                        InputShapesMap max_inputs_shape,
                        InputDataTypeMap inputs_data_type,
                        bool enable_const_folder=true);

    // @brief network deinit to release init create resource
    virtual Status DeInit();

    // @brief reshape with input shape info
    // @inputs input shape info
    virtual Status Reshape(const InputShapesMap &inputs);

    // @brief network infer, it will sync to wait result
    virtual Status Forward();

    // @brief tnn instance network infer, it will not wait
    virtual Status ForwardAsync(Callback call_back);
    
    //  @brief return the amount of memory required for forward
    //  @param memory_size: the memory size used by tnn layers for forward
    //  @return error code: If successful, returns zero. Otherwise, return an error code.
    virtual Status GetForwardMemorySize(int &memory_size);

    //  @brief: set memory used by the tnn instance without forward
    //  memory, the memory size must be at least that returned by
    //  GetForwardMemorySize(). releasing or otherwise using the memory for
    //  other purposes during the tnn network run will result in
    //  undefined behavior.
    //  @param memory: the memory used by tnn layers for forward
    //  @return error code: If successful, returns zero. Otherwise, return an error code.
    virtual Status SetForwardMemory(void *memory);

    // @brief get tnn command queue
    // @param command_queue device command queue for forward
    //virtual Status GetCommandQueue(void **command_queue);

    // @brief get all input blobs
    // @param blobs input blobs name map
    virtual Status GetAllInputBlobs(BlobMap &blobs);

    // @brief get all output blobs
    // @param blobs output blobs name map
    virtual Status GetAllOutputBlobs(BlobMap &blobs);

private:
    std::unique_ptr<zdl::SNPE::SNPE> snpe_;
    zdl::DlSystem::UserBufferMap input_map_;
    zdl::DlSystem::UserBufferMap output_map_;
    std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>> snpe_userbacked_input_buffers_;
    std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>> snpe_userbacked_output_buffers_;
    std::unordered_map<std::string, std::vector<uint8_t>> application_input_buffers_;
    std::unordered_map<std::string, std::vector<uint8_t>> application_output_buffers_;

    BlobMap input_blob_map_;
    BlobMap output_blob_map_;
};

}  // namespace TNN_NS
#endif  // TNN_SOURCE_TNN_DEVICE_SNPE_SNPE_NETWORK_H_
