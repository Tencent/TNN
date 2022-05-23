// Copyright 2019 Tencent. All Rights Reserved

#include "snpe_network.h"
#include "DlSystem/DlEnums.hpp"
#include "DlSystem/DlVersion.hpp"
#include "DlSystem/String.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "snpe_model_interpreter.h"
#include "snpe_utils.h"

namespace TNN_NS {

NetworkImplFactoryRegister<NetworkImplFactory<SnpeNetwork>>
    g_network_impl_snpe_factory_register(NETWORK_TYPE_SNPE);

SnpeNetwork::~SnpeNetwork() {
    DeInit();
}

Status SnpeNetwork::Init(NetworkConfig &net_config, ModelConfig &model_config,
                         AbstractModelInterpreter *interpreter,
                         InputShapesMap inputs_shape) {
    SnpeModelInterpreter *snpe_interpreter =
        dynamic_cast<SnpeModelInterpreter *>(interpreter);

    std::unique_ptr<zdl::DlContainer::IDlContainer> &container =
        snpe_interpreter->GetContainer();

    zdl::DlSystem::Version_t version =
        zdl::SNPE::SNPEFactory::getLibraryVersion();
    LOGD("SPNE Version: %s\n", version.asString().c_str());

    zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::DSP;
    if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime)) {
        LOGE("DSP Runtime not avaliable!\n");
        return TNNERR_DEVICE_NOT_SUPPORT;
    }

    zdl::DlSystem::PlatformConfig platform_config;
    zdl::DlSystem::RuntimeList runtime_list;
    zdl::DlSystem::UDLFactoryFunc udl_func = nullptr;  // ?? check
    zdl::DlSystem::UDLBundle udlbundle;
    udlbundle.cookie = (void *)0xdeadbeaf;
    udlbundle.func   = udl_func;  // 0xdeadbeaf to test cookie

    snpe_ = SetBuilderOptions(container, runtime, runtime_list, udlbundle, true,
                              platform_config, false);
    if (snpe_ == nullptr) {
        LOGE("Build snpe falied!\n");
        return TNNERR_DEVICE_NOT_SUPPORT;
    }

    CreateInputBufferMap(input_map_, input_blob_map_,
                         application_input_buffers_,
                         snpe_userbacked_input_buffers_, snpe_, false);
    CreateOutputBufferMap(output_map_, output_blob_map_,
                          application_output_buffers_,
                          snpe_userbacked_output_buffers_, snpe_, false);

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
    LOGE("Snpe Reshape!\n");
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

Status SnpeNetwork::GetCommandQueue(void **command_queue) {
    return TNN_OK;
}

Status SnpeNetwork::Forward() {
    LOGE("Snpe Forward!\n");
    bool ret = snpe_->execute(input_map_, output_map_);
    if (ret)
        return TNN_OK;
    else
        return TNNERR_SNPE_API_ERROR;
}

Status SnpeNetwork::ForwardAsync(Callback call_back) {
    LOGE("Snpe Async Forward! (as same as Forward by now)\n");
    bool ret = snpe_->execute(input_map_, output_map_);
    if (ret)
        return TNN_OK;
    else
        return TNNERR_SNPE_API_ERROR;
}

}  // namespace TNN_NS
