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

#include "tnn/device/snpe/snpe_dsp_context.h"

namespace TNN_NS {

Status SnpeDspContext::LoadLibrary(std::vector<std::string> path) {
    return TNN_OK;
}

Status SnpeDspContext::GetCommandQueue(void** command_queue) {
    return TNN_OK;
}

Status SnpeDspContext::SetCommandQueue(void* command_queue) {
    return TNN_OK;
}

Status SnpeDspContext::ShareCommandQueue(Context* context) {
    return TNN_OK;
}

Status SnpeDspContext::OnInstanceForwardBegin() {
    Context::OnInstanceForwardBegin();
    return TNN_OK;
}

Status SnpeDspContext::SetNumThreads(int num_threads) {
    return TNN_OK;
}

int SnpeDspContext::GetNumThreads() {
    return 1;
}

Status SnpeDspContext::OnInstanceForwardEnd() {
    return TNN_OK;
}

Status SnpeDspContext::Synchronize() {
    return TNN_OK;
}

}  // namespace TNN_NS
