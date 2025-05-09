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

#include "DlSystem/DlEnums.hpp"
#include "DlSystem/DlVersion.hpp"
#include "DlSystem/String.hpp"
#include "SNPE/SNPEFactory.hpp"

#include "tnn/device/snpe/snpe_model_interpreter.h"
#include "tnn/device/snpe/snpe_network.h"
#include "tnn/device/snpe/snpe_utils.h"

namespace TNN_NS {

NetworkImplFactoryRegister<NetworkImplFactory<SnpeNetwork>>
    g_network_impl_snpe_factory_register(NETWORK_TYPE_SNPE);

SnpeNetwork::~SnpeNetwork() {
    DeInit();
}

Status SnpeNetwork::Init(NetworkConfig &net_config, ModelConfig &model_config,
                         AbstractModelInterpreter *interpreter,
                         InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape,
                         InputDataTypeMap inputs_data_type, bool enable_const_folder) {
    SnpeModelInterpreter *snpe_interpreter =
        dynamic_cast<SnpeModelInterpreter *>(interpreter);

    std::unique_ptr<zdl::DlContainer::IDlContainer> &container =
        snpe_interpreter->GetContainer();

    zdl::DlSystem::Version_t version =
        zdl::SNPE::SNPEFactory::getLibraryVersion();
    LOGD("Run TNN SNPE with SPNE Version: %s\n", version.asString().c_str());
    
    zdl::DlSystem::Runtime_t runtime = SelectSNPERuntime();
    if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime)) {
        LOGE("SNPE Runtime not avaliable!\n");
        return TNNERR_DEVICE_NOT_SUPPORT;
    }

    //LoadUdoPackages();
    
    zdl::DlSystem::PlatformConfig platform_config;
    zdl::DlSystem::RuntimeList runtime_list;
    zdl::DlSystem::StringList outputs;
    for (int i = 1; i < model_config.params.size(); i++) {
        outputs.append(model_config.params[i].c_str());
    }

    snpe_ = SetBuilderOptions(container, runtime, runtime_list, true,
                              platform_config, false, outputs);
    if (snpe_ == nullptr) {
        LOGE("Build snpe falied, API SetBuilderOptions() return nullptr.\n");
        return TNNERR_DEVICE_NOT_SUPPORT;
    }

    CreateInputBufferMap(input_map_, input_blob_map_,
                         application_input_buffers_,
                         snpe_userbacked_input_buffers_, snpe_, false);
    CreateOutputBufferMap(output_map_, output_blob_map_,
                          application_output_buffers_,
                          snpe_userbacked_output_buffers_, snpe_, false);

    device_ = GetDevice(DEVICE_DSP);
    if (device_ == NULL) {
        return TNNERR_DEVICE_NOT_SUPPORT;
    }

    context_ = device_->CreateContext(net_config.device_id);
    context_->LoadLibrary(net_config.library_path);

    return TNN_OK;
}

Status SnpeNetwork::GetForwardMemorySize(int &memory_size) {
    memory_size = 0;
    return TNN_OK;
}

Status SnpeNetwork::SetForwardMemory(void *memory) {
    return TNN_OK;
}

Status SnpeNetwork::GetAllInputBlobs(BlobMap &blobs) {
    blobs = input_blob_map_;
    return TNN_OK;
}

Status SnpeNetwork::GetAllOutputBlobs(BlobMap &blobs) {
    blobs = output_blob_map_;
    return TNN_OK;
}

Status SnpeNetwork::Reshape(const InputShapesMap &inputs) {
    LOGD("Calling TNN SNPE Network Reshape\n");
    return TNN_OK;
}

Status SnpeNetwork::DeInit() {
    for (auto item : input_blob_map_) {
        delete item.second;
    }
    input_blob_map_.clear();
    for (auto item : output_blob_map_) {
        delete item.second;
    }
    output_blob_map_.clear();
    return TNN_OK;
}

//Status SnpeNetwork::GetCommandQueue(void **command_queue) {
//    return TNN_OK;
//}

Status SnpeNetwork::Forward() {
    LOGD("Calling TNN SNPE Network Forward\n");
    bool ret = snpe_->execute(input_map_, output_map_);
    if (!ret) {
        LOGE("TNN SnpeNetwork::Forward returned non-zero.\n");
        return TNNERR_SNPE_API_ERROR;
    }
    return TNN_OK;
}

Status SnpeNetwork::ForwardAsync(Callback call_back) {
    LOGD("Calling TNN SNPE Network ForwardAsync\n");
    bool ret = snpe_->execute(input_map_, output_map_);
    if (!ret) {
        LOGE("TNN SnpeNetwork::ForwardAsync returned non-zero.\n");
        return TNNERR_SNPE_API_ERROR;
    }
    return TNN_OK;
}

}  // namespace TNN_NS
