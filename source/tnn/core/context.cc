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

#include "tnn/core/context.h"
#include "tnn/core/profile.h"
#include "tnn/utils/string_format.h"

namespace TNN_NS {

// this function is called before forward by Network.
Status Context::OnInstanceForwardBegin() {
    return TNN_OK;
}

// this function is called before Reshape by Network.
Status Context::OnInstanceReshapeBegin() {
    return TNN_OK;
}

// this function is called after Reshape by Network.
Status Context::OnInstanceReshapeEnd() {
    return TNN_OK;
}

Status Context::ShareCommandQueue(Context* context) {
    LOGE("Subclass of Context must implement this func SetCommandQueue\n");
    return Status(TNNERR_COMMON_ERROR, "Subclass of Context must implement this func SetCommandQueue");
}

/*
 * Implement by the actual context such as ArmContext etc.
 * Not implemented for this default context.
 */
Status Context::SetNumThreads(int num_threads) {
    return TNN_OK;
}

void Context::SetPrecision(Precision precision) {
    precision_ = precision;
}

Precision Context::GetPrecision() {
    return precision_;
}

void Context::SetEnableTuneKernel(bool enable_tune_kernel) {
    enable_tune_kernel_ = enable_tune_kernel;
}

bool Context::GetEnableTuneKernel() {
    return enable_tune_kernel_;
}

#if TNN_PROFILE
void Context::StartProfile() {
    profile_layer     = true;
    profiling_result_ = std::make_shared<ProfileResult>();
}

std::shared_ptr<ProfileResult> Context::FinishProfile() {
    profile_layer = false;
    return profiling_result_;
}

void Context::AddProfilingData(std::shared_ptr<ProfilingData> pdata) {
    if (profile_layer && profiling_result_) {
        profiling_result_->AddProfilingData(pdata);
    }
}
#endif

}  // namespace TNN_NS
