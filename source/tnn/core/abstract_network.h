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

#ifndef TNN_SOURCE_TNN_CORE_ABSTRACT_NETWORK_H_
#define TNN_SOURCE_TNN_CORE_ABSTRACT_NETWORK_H_

#include <map>
#include <memory>
#include <vector>

#include "tnn/core/blob.h"
#include "tnn/core/common.h"
#include "tnn/core/instance.h"
#include "tnn/core/macro.h"
#include "tnn/core/profile.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/abstract_model_interpreter.h"

namespace TNN_NS {

class AbstractNetwork {
public:
    // @brief virtual default destructor
    virtual ~AbstractNetwork() {}

    // @brief init network with net cfg and net res.
    // @param net_cfg
    // @param net_res
    virtual Status Init(NetworkConfig &net_config, ModelConfig &model_config, AbstractModelInterpreter *interpreter,
                        InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape) = 0;

    // @brief deinit release init create resource
    virtual Status DeInit() = 0;

    //  @brief return the amount of memory required for forward
    //  @param memory_size: the memory size used by tnn layers for
    //  forward
    //  @return error code: If successful, returns zero. Otherwise, returns
    //  an error code.
    virtual Status GetForwardMemorySize(int &memory_size) = 0;

    //  @brief: set memory used by the tnn instance without forward
    //  memory, the memory size must be at least that returned by
    //  GetForwardMemorySize(). releasing or otherwise using the memory for
    //  other purposes during the tnn network run will result in
    //  undefined behavior.
    //  @param memory: the memory used by tnn layers for forward
    //  @return error code: If successful, returns zero. Otherwise, returns
    //  an error code.
    //
    virtual Status SetForwardMemory(void *memory) = 0;

    // @brief network infer
    virtual Status Reshape(const InputShapesMap &inputs) = 0;

    // @brief get tnn command queue
    // @param command_queue device command queue for forward
    virtual Status GetCommandQueue(void **command_queue) = 0;

    // @brief share tnn command queue to another network。
    virtual Status ShareCommandQueue(AbstractNetwork *network);
    
    // @brief network infer, it will sync to wait result
    virtual Status Forward() = 0;

#ifdef FORWARD_CALLBACK_ENABLE
    // @brief network infer with callbach to statistic blob info
    virtual Status ForwardWithCallback(BlobStatisticCallback before, BlobStatisticCallback after);
#endif  // end of FORWARD_CALLBACK_ENABLE

    // @brief tnn instance network infer, it will not wait
    virtual Status ForwardAsync(Callback call_back) = 0;

    // @brief get all input blobs
    // @param blobs input blobs name map
    virtual Status GetAllInputBlobs(BlobMap &blobs) = 0;

    // @brief get all output blobs
    // @param blobs output blobs name map
    virtual Status GetAllOutputBlobs(BlobMap &blobs) = 0;

    // @brief set threads run on device
    virtual Status SetCpuNumThreads(int num_threads);

#if TNN_PROFILE
public:
    virtual void StartProfile();
    virtual std::shared_ptr<ProfileResult> FinishProfile();
#endif
};

class AbstractNetworkImplFactory {
public:
    virtual ~AbstractNetworkImplFactory() {}
    virtual std::shared_ptr<AbstractNetwork> CreateNetworkImp() = 0;
};

template <typename T>
class NetworkImplFactory : public AbstractNetworkImplFactory {
public:
    virtual std::shared_ptr<AbstractNetwork> CreateNetworkImp() {
        return std::make_shared<T>();
    }
};

class NetworkImplManager {
public:
    static std::shared_ptr<AbstractNetwork> GetNetworkImpl(NetworkType type);
    static void RegisterNetworkImplFactory(NetworkType type, AbstractNetworkImplFactory *factory);

private:
    static std::map<NetworkType, std::shared_ptr<AbstractNetworkImplFactory>> &GetNetworkImplFactoryMap();
};

template <typename T>
class NetworkImplFactoryRegister {
public:
    explicit NetworkImplFactoryRegister(NetworkType type) {
        NetworkImplManager::RegisterNetworkImplFactory(type, new T());
    }
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_CORE_ABSTRACT_NETWORK_H_
