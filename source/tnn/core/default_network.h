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

#ifndef TNN_SOURCE_TNN_CORE_DEFAULT_NETWORK_H_
#define TNN_SOURCE_TNN_CORE_DEFAULT_NETWORK_H_

#include <vector>

#include "tnn/core/abstract_device.h"
#include "tnn/core/abstract_network.h"
#include "tnn/core/blob.h"
#include "tnn/core/blob_manager.h"
#include "tnn/core/common.h"
#include "tnn/core/context.h"
#include "tnn/core/macro.h"
#include "tnn/core/profile.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/abstract_model_interpreter.h"
#include "tnn/interpreter/layer_resource.h"
#include "tnn/interpreter/net_resource.h"
#include "tnn/interpreter/net_structure.h"
#include "tnn/layer/base_layer.h"

namespace TNN_NS {

class DefaultNetwork : public AbstractNetwork {
public:
    // @brief DefaultNetwork Constructor
    DefaultNetwork();

    // @brief DefaultNetwork virtual Destructor
    virtual ~DefaultNetwork();

public:
    // @brief int net with network config, net structure and net resource info
    // @param config network config info
    // @param net_structure network structure info
    // @param net_resource network resource info
    // @param inputs_shape_map modify input shape, if empty, it will use the
    // shape in proto
    virtual Status Init(NetworkConfig &net_config, ModelConfig &model_config, AbstractModelInterpreter *interpreter,
                        InputShapesMap inputs_shape);

    // @brief reshape with input shape info
    // @inputs input shape info
    virtual Status Reshape(const InputShapesMap &inputs);

    // @brief get tnn command queue
    // @param command_queue device command queue for forward
    virtual Status GetCommandQueue(void **command_queue);

    // @brief network forward
    virtual Status Forward();

#ifdef FORWARD_CALLBACK_ENABLE
    // @brief network infer with callbach to statistic blob info
    virtual Status ForwardWithCallback(BlobStatisticCallback before, BlobStatisticCallback after);
#endif  // end of FORWARD_CALLBACK_ENABLE

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

    // @brief set threads run on device
    virtual Status SetCpuNumThreads(int num_threads);

#if TNN_PROFILE
public:
    virtual void StartProfile();
    virtual std::shared_ptr<ProfileResult> FinishProfile();
#endif

private:
    virtual Status InitLayers(NetStructure *net_structure, NetResource *net_resource);
    Status GenerateInt8Blob(const std::string &name, NetResource *net_resource, Blob **blob);
    Status UpdateBlobPrecision(std::shared_ptr<LayerInfo> layer_info, bool is_input, bool is_quantized_net,
                               const std::string &name, NetResource *net_resource, Blob **blob);

    AbstractDevice *device_ = nullptr;
    Context *context_       = nullptr;

    std::vector<BaseLayer *> layers_;

    BlobManager *blob_manager_ = nullptr;

    NetStructure *net_structure_ = nullptr;

    NetworkConfig config_;

    static std::mutex optimize_mtx_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_CORE_DEFAULT_NETWORK_H_
