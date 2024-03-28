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

#include "tnn/device/atlas/atlas_runtime.h"
#include "acl/acl.h"
#include "tnn/core/macro.h"

namespace TNN_NS {

static std::mutex g_mtx;

std::shared_ptr<AtlasRuntime> AtlasRuntime::atlas_runtime_singleton_ = nullptr;
int AtlasRuntime::ref_count_                                         = 0;
bool AtlasRuntime::init_done_                                        = false;

AtlasRuntime* AtlasRuntime::GetInstance() {
    if (nullptr == atlas_runtime_singleton_.get()) {
        std::unique_lock<std::mutex> lck(g_mtx);
        if (nullptr == atlas_runtime_singleton_.get()) {
            atlas_runtime_singleton_.reset(new AtlasRuntime());
        }
    }

    return atlas_runtime_singleton_.get();
}

void AtlasRuntime::DecreaseRef() {
    std::unique_lock<std::mutex> lck(g_mtx);
    ref_count_--;
    LOGD("AtlasRuntime::DecreaseRef() count=%d\n", ref_count_);
    if (ref_count_ <= 0) {
        atlas_runtime_singleton_.reset();
        ref_count_ = 0;
    }
}

AtlasRuntime::AtlasRuntime() {
    device_list_.clear();
}

// Init Atlas Runtime and increase reference count
Status AtlasRuntime::Init() {
    std::unique_lock<std::mutex> lck(g_mtx);

    ref_count_++;
    LOGD("AtlasRuntime::Init() reference count=%d\n", ref_count_);

    // only init once.
    if (!init_done_) {
        LOGD("Init Atlas Acl\n");

        LOGD("acl begin init...\n");
        aclError ret = aclInit(nullptr);
        if (ret != ACL_ERROR_NONE && ret != ACL_ERROR_REPEAT_INITIALIZE) {
            LOGE("acl init failed!\n");
            return TNNERR_ATLAS_RUNTIME_ERROR;
        }
        LOGD("acl init done!\n");

        init_done_ = true;
    }

    return TNN_OK;
}

Status AtlasRuntime::SetDevice(int device_id) {
    std::unique_lock<std::mutex> lck(g_mtx);
    if (device_list_.find(device_id) == device_list_.end()) {
        LOGD("set device: %d\n", device_id);
        aclError acl_ret = aclrtSetDevice(device_id);
        if (acl_ret != ACL_ERROR_NONE) {
            LOGE("acl open device %d failed (acl error code: %d)\n", device_id, acl_ret);
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "acl open device falied");
        }
        LOGD("set device done!\n");
        device_list_.emplace(device_id);
    }

    return TNN_OK;
}

Status AtlasRuntime::AddModelInfo(Blob* blob, AtlasModelInfo model_info) {
    std::unique_lock<std::mutex> lck(g_mtx);
    model_info_map_[blob] = model_info;
    return TNN_OK;
}

Status AtlasRuntime::DelModelInfo(Blob* blob) {
    std::unique_lock<std::mutex> lck(g_mtx);
    auto blob_it = model_info_map_.find(blob);
    if (blob_it != model_info_map_.end()) {
        model_info_map_.erase(blob_it);
    }
    return TNN_OK;
}

std::map<Blob*, AtlasModelInfo>& AtlasRuntime::GetModleInfoMap() {
    return model_info_map_;
}

AtlasRuntime::~AtlasRuntime() {
    LOGD("~AtlasRuntime() begin \n");

    aclError ret;
    for (auto id : device_list_) {
        LOGD("reset device: %d\n", id);
        ret = aclrtResetDevice(id);
        if (ret != ACL_ERROR_NONE) {
            LOGE("acl reset device failed!\n");
        }
    }
    device_list_.clear();

    LOGD("aclFinalize()\n");
    ret = aclFinalize();
    if (ret != ACL_ERROR_NONE) {
        LOGD("acl finalize failed!\n");
    }

    LOGD("~AtlasRuntime() end \n");
}

}  // namespace TNN_NS
