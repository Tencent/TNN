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

#include "tnn/core/abstract_network.h"

#include "tnn/core/profile.h"

namespace TNN_NS {

#ifdef FORWARD_CALLBACK_ENABLE
Status AbstractNetwork::ForwardWithCallback(BlobStatisticCallback before, BlobStatisticCallback after) {
    return TNN_OK;
}
#endif  // end of FORWARD_CALLBACK_ENABLE

Status AbstractNetwork::ShareCommandQueue(AbstractNetwork *network) {
    LOGE("Subclass of AbstractNetwork must implement this func ShareCommandQueue\n");
    return Status(TNNERR_COMMON_ERROR, "Subclass of AbstractNetwork must implement this func ShareCommandQueue");
}

Status AbstractNetwork::ShareNetResource(AbstractNetwork *network) {
    LOGE("Subclass of AbstractNetwork must implement this func ShareNetResource\n");
    return Status(TNNERR_COMMON_ERROR, "Subclass of AbstractNetwork must implement this func ShareNetResource");
}

Status AbstractNetwork::SetCpuNumThreads(int num_threads) {
    return TNN_OK;
}

#if TNN_PROFILE
void AbstractNetwork::StartProfile() {
    LOGI("warning: to make profiling work, subclass should implement the func: StartProfile\n");
}

std::shared_ptr<ProfileResult> AbstractNetwork::FinishProfile() {
    LOGI("warning: to make profiling work, subclass should implement the func: FinishProfile\n");
    return std::make_shared<ProfileResult>();
}
#endif

std::map<NetworkType, std::shared_ptr<AbstractNetworkImplFactory>> &NetworkImplManager::GetNetworkImplFactoryMap() {
    static std::map<NetworkType, std::shared_ptr<AbstractNetworkImplFactory>> s_network_impl_factory_map;
    return s_network_impl_factory_map;
}

std::shared_ptr<AbstractNetwork> NetworkImplManager::GetNetworkImpl(NetworkType type) {
    auto &impl_map = NetworkImplManager::GetNetworkImplFactoryMap();
    auto iter      = impl_map.find(type);
    if (iter != impl_map.end()) {
        return iter->second->CreateNetworkImp();
    }

    return nullptr;
}

/*
 * NetworkImpl is registered the map at runtime.
 */
void NetworkImplManager::RegisterNetworkImplFactory(NetworkType type, AbstractNetworkImplFactory *factory) {
    if (factory) {
        auto &impl_map = NetworkImplManager::GetNetworkImplFactoryMap();
        impl_map[type] = std::shared_ptr<AbstractNetworkImplFactory>(factory);
    }
}


Status AbstractNetwork::InitWrapper(NetworkConfig &net_config, ModelConfig &model_config, AbstractModelInterpreter *interpreter,
        InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape, InputDataTypeMap inputs_data_type, bool enable_const_folder) {
    Status ret = Init(net_config, model_config, interpreter, min_inputs_shape, max_inputs_shape, inputs_data_type, enable_const_folder);
    if(ret != TNN_OK) {
        return ret;
    }

    BlobMap inputs;
    ret = GetAllInputBlobs(inputs);
    if (ret != TNN_OK) {
        LOGE("ERROR: get input blobs failed");
        return ret;
    }

    // init min max shapes
    for(auto iter : inputs) {
        max_inputs_shape_[iter.first] = iter.second->GetBlobDesc().dims;
        if(min_inputs_shape.count(iter.first) > 0) {
            min_inputs_shape_[iter.first] = min_inputs_shape[iter.first];
        } else {
            min_inputs_shape_[iter.first] = iter.second->GetBlobDesc().dims;
        }
    }

    return ret;
}

Status AbstractNetwork::ReshapeWrapper(const InputShapesMap &inputs) {
    for(auto iter : inputs) {
        auto name = iter.first;
        auto dims = iter.second;
        if(min_inputs_shape_.count(name) > 0) {
            auto min_dims = min_inputs_shape_[name];
            auto max_dims = max_inputs_shape_[name];
            if(min_dims.size() != dims.size()) {
                return Status(TNNERR_PARAM_ERR, "input shape dims error \n");
            } else {
                for(int i = 0; i < dims.size(); ++i) {
                    if((dims[i] > max_dims[i])) {
                        return Status(TNNERR_PARAM_ERR, "input shape dims error \n");
                    }
                }
            }
        } else {
            return Status(TNNERR_PARAM_ERR, "input shape dims error \n");
        }
    }

    return Reshape(inputs);
}

}  // namespace TNN_NS
