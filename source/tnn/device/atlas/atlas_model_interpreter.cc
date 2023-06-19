// Copyright 2019 Tencent. All Rights Reserved

#include "tnn/device/atlas/atlas_model_interpreter.h"

#include <fstream>
#include "tnn/device/atlas/atlas_utils.h"
#include "tnn/device/atlas/atlas_runtime.h"
#include "tnn/utils/split_utils.h"

namespace TNN_NS {

AtlasModelInterpreter::AtlasModelInterpreter() {}

AtlasModelInterpreter::~AtlasModelInterpreter() {
    LOGD("~AtlasModelInterpreter()\n");
    for (auto &item : model_weight_map_) {
        if (nullptr != item.second.weight_mem_ptr && nullptr != item.second.context) {
            aclError ret = aclrtSetCurrentContext(item.second.context);
            if (ret != ACL_ERROR_NONE) {
                LOGE("set context failed\n");
            }

            aclrtFree(item.second.weight_mem_ptr);
            LOGD("acl free model weight ptr (device: %d)\n", item.first);
            item.second.weight_mem_ptr = nullptr;

            ret = aclrtDestroyContext(item.second.context);
            if (ret != ACL_ERROR_NONE) {
                LOGE("destroy context failed\n");
            }
            item.second.context = nullptr;
        }
    }
    model_weight_map_.clear();
    model_weight_size_ = 0;
    AtlasRuntime::DecreaseRef();
}

Status AtlasModelInterpreter::Interpret(std::vector<std::string> &params) {
    model_config_.om_str  = params[0];
    model_config_.is_path = false;
    if (model_config_.om_str.length() < 1024) {
        std::ifstream om_file(model_config_.om_str);
        if (!om_file) {
            LOGE("Invalied om file path! (param[0] : %s) take as memory content\n", model_config_.om_str.c_str());
            model_config_.is_path = false;
        } else {
            model_config_.is_path = true;
        }
    }

    // Init ACL
    Status tnn_ret = AtlasRuntime::GetInstance()->Init();
    if (tnn_ret != TNN_OK) {
        LOGE("acl init falied\n");
        return tnn_ret;
    }

    size_t model_mem_size;
    aclError acl_ret = ACL_ERROR_NONE;
    if (model_config_.is_path) {
        acl_ret = aclmdlQuerySize(model_config_.om_str.c_str(), &model_mem_size, &model_weight_size_);
    } else {
        acl_ret = aclmdlQuerySizeFromMem(model_config_.om_str.data(), model_config_.om_str.length(), &model_mem_size,
                                         &model_weight_size_);
    }
    if (acl_ret != ACL_ERROR_NONE) {
        LOGE("query model failed (%d)\n", acl_ret);
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "query model failed");
    }
    LOGD("atlas model weight size: %d  model mem size: %d\n", model_weight_size_, model_mem_size);

    return TNN_OK;
}

AtlasModelConfig& AtlasModelInterpreter::GetModelConfig() {
    return model_config_;
}

void* AtlasModelInterpreter::GetModelWeightsBufferPtr(int device_id) {
    std::unique_lock<std::mutex> lck(mutex_);
    if (model_weight_map_.find(device_id) == model_weight_map_.end()) {
        WeightPacket packet;
        // create context related to device
        aclError acl_ret = aclrtCreateContext(&packet.context, device_id);
        if (acl_ret != ACL_ERROR_NONE) {
            LOGE("acl create context failed (device %d) (acl error code: %d)\n", device_id, acl_ret);
            return nullptr;
        }

        // alloc device memory
        acl_ret = aclrtMalloc(&packet.weight_mem_ptr, model_weight_size_, ACL_MEM_MALLOC_HUGE_FIRST);
        if (acl_ret != ACL_ERROR_NONE) {
            LOGE("malloc buffer for weight failed (ret=%d), require size is %zu\n", acl_ret, model_weight_size_);
            return nullptr;
        }
        LOGD("malloc buffer for weight success (size %zu)\n", model_weight_size_);

        model_weight_map_[device_id] = packet;
    }

    return model_weight_map_[device_id].weight_mem_ptr;
}

size_t AtlasModelInterpreter::GetModelWeightsBufferSize() {
    return model_weight_size_;
}

TypeModelInterpreterRegister<TypeModelInterpreterCreator<AtlasModelInterpreter>> g_atlas_model_interpreter_register(
    MODEL_TYPE_ATLAS);

}  // namespace TNN_NS
