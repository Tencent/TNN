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

#include "tnn/device/atlas/atlas_context.h"

namespace TNN_NS {

AtlasContext::~AtlasContext() {
    // Aclrt Stream is created and maintained by AtlasNetwork
    // Do not Destroy aclrtStream HERE.
    //if (this->aclrt_stream_ != nullptr) {
    //    ret = aclrtDestroyStream(this->aclrt_stream_);
    //    this->aclrt_stream_ = nullptr;
    //}
}

Status AtlasContext::LoadLibrary(std::vector<std::string> path) {
    return TNN_OK;
}

Status AtlasContext::GetCommandQueue(void** command_queue) {
    // Reshape Model For different Model Types
    if (this->model_type_ == MODEL_TYPE_TORCHSCRIPT) {
        LOGE("Fail to GetCommandQueue, MODEL_TYPE_TORCHSCRIPT not supported YET.\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "Fail to GetCommandQueue, MODEL_TYPE_TORCHSCRIPT not supported YET");
    } else if (this->model_type_ == MODEL_TYPE_TNN || this->model_type_ == MODEL_TYPE_RAPIDNET) {
        LOGE("Fail to GetCommandQueue, MODEL_TYPE_TNN not supported YET.\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "Fail to GetCommandQueue, MODEL_TYPE_TNN not supported YET");
    } else if (this->model_type_ == MODEL_TYPE_ATLAS) {
        *command_queue = this->aclrt_stream_;
    } else {
        LOGE("Fail to GetCommandQueue, model type not supported.\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "Fail to GetCommandQueue, model type not supported");
    }

    return TNN_OK;
}

Status AtlasContext::SetCommandQueue(void* command_queue) {
    return TNN_OK;
}

Status AtlasContext::ShareCommandQueue(Context* context) {
    return TNN_OK;
}

Status AtlasContext::OnInstanceForwardBegin() {
    return TNN_OK;
}

Status AtlasContext::OnInstanceForwardEnd() {
    return TNN_OK;
}

Status AtlasContext::Synchronize() {
    if (model_type_ == MODEL_TYPE_TNN || model_type_ == MODEL_TYPE_RAPIDNET ||
        model_type_ == MODEL_TYPE_ATLAS) {
        aclError acl_ret = aclrtSynchronizeStream(this->aclrt_stream_);
        if (acl_ret != ACL_ERROR_NONE) {
            LOGE("before forward synchronize stream failed\n");
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "before forward synchronize stream failed");
        }
    }
    return TNN_OK;
}

aclrtStream& AtlasContext::GetAclrtStream() {
    return this->aclrt_stream_;
}

void AtlasContext::SetAclrtStream(const aclrtStream& stream) {
    this->aclrt_stream_ = stream;
}

Status AtlasContext::CreateAclrtStream() {
    // Create aclrt Stream
    aclError acl_ret = aclrtCreateStream(&aclrt_stream_);
    if (acl_ret != ACL_ERROR_NONE) {
        LOGE("acl create stream failed (acl error code: %d)\n", acl_ret);
    }
    return TNN_OK;
}

ModelType& AtlasContext::GetModelType() {
    return this->model_type_;
}

void AtlasContext::SetModelType(ModelType model_type) {
    this->model_type_ = model_type;
}

void AtlasContext::SetDeviceId(int device_id) {
    this->device_id_ = device_id;
}

int AtlasContext::GetDeviceId() {
    return this->device_id_;
}

}  //  namespace TNN_NS
