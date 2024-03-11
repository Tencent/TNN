// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef TNN_SOURCE_TNN_NETWORK_TORCHAIE_TORCHAIE_NETWORK_H_
#define TNN_SOURCE_TNN_NETWORK_TORCHAIE_TORCHAIE_NETWORK_H_

#include <iostream>
#include <unordered_set>
#include <thread>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/jit/api/module.h>

#include "tnn/core/abstract_network.h"

namespace TNN_NS {

class TorchAieNetwork: public AbstractNetwork {
public:

    // @brief virtual default destructor
    virtual ~TorchAieNetwork();

    // @brief int net with network config, net structure and net resource info
    // @param config network config info
    // @param net_structure network structure info
    // @param net_resource network resource info
    // @param inputs_shape_map modify input shape, if empty, it will use the
    // shape in proto
	// @param inputs_data_type modify input data type, by default float.
    virtual Status Init(
        NetworkConfig &net_config,
        ModelConfig &model_config,
        AbstractModelInterpreter* interpreter,
        InputShapesMap min_inputs_shape,
        InputShapesMap max_inputs_shape,
        InputDataTypeMap inputs_data_type,
        bool enable_const_folder);

    // @brief deinit release init create resource
    virtual Status DeInit();

    //  @brief return the amount of memory required for forward
    //  @param memory_size: the memory size used by tnn layers for
    //  forward
    //  @return error code: If successful, returns zero. Otherwise, returns
    //  an error code.
    virtual Status GetForwardMemorySize(size_t &memory_size);

    //  @brief: set memory used by the tnn instance without forward
    //  memory, the memory size must be at least that returned by
    //  GetForwardMemorySize(). releasing or otherwise using the memory for
    //  other purposes during the tnn network run will result in
    //  undefined behavior.
    //  @param memory: the memory used by tnn layers for forward
    //  @return error code: If successful, returns zero. Otherwise, returns
    //  an error code.
    //
    virtual Status SetForwardMemory(void *memory);

    // @brief get tnn command queue
    // @param command_queue device command queue for forward
    virtual Status GetCommandQueue(void **command_queue);

    // @brief set tnn command queue
    virtual Status SetCommandQueue(void *command_queue) { return TNN_OK; }

    // @brief network infer
    virtual Status Reshape(const InputShapesMap &inputs);

    // @brief get all input blobs
    // @param blobs input blobs name map
    virtual Status GetAllInputBlobs(BlobMap &blobs);

    // @brief get all output blobs
    // @param blobs output blobs name map
    virtual Status GetAllOutputBlobs(BlobMap &blobs);

    // @brief network forward
    virtual Status Forward();

    // @brief tnn instance network infer, it will not wait
    virtual Status ForwardAsync(Callback call_back);

    // @brief set input tensor
    Status SetInputTensor(at::Tensor tensor, std::string name);

    // @brief get output tensor
    at::Tensor GetOutputTensor(std::string name);

private:
    std::shared_ptr<torch::jit::Module> Compile(
        InputShapesMap &min_inputs_shape,
        InputShapesMap &max_inputs_shape,
        InputDataTypeMap &inputs_data_type,
        torch::jit::Module &module);

    Status UpdateInputBlobMap(InputShapesMap &inputs_shape, InputDataTypeMap &inputs_data_type);
    Status UpdateOutputBlobMap();
    at::Tensor ConvertBlobToTensor(Blob *blob);

    NetworkConfig network_config_;
    BlobMap input_blob_map_;
    BlobMap output_blob_map_;
    std::map<std::string, at::Tensor> input_tensors_;
    std::map<std::string, at::Tensor> output_tensors_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::shared_ptr<torch::jit::Module> aie_module_ = nullptr;
};

}  //  namespace TNN_NS

#endif  //  TNN_SOURCE_TNN_NETWORK_TORCHAIE_TORCHAIE_NETWORK_H_
