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

#include "tnn/device/arm/arm_context.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/utils/cpu_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

Status ArmContext::LoadLibrary(std::vector<std::string> path) {
    return TNN_OK;
}

Status ArmContext::GetCommandQueue(void** command_queue) {
    return TNN_OK;
}

Status ArmContext::ShareCommandQueue(Context* context) {
    return TNN_OK;
}

Status ArmContext::OnInstanceForwardBegin() {
    Context::OnInstanceForwardBegin();
    OMP_SET_THREADS_(GetNumThreads());
    return TNN_OK;
}

Status ArmContext::OnInstanceForwardEnd() {
    return TNN_OK;
}

Status ArmContext::Synchronize() {
    return TNN_OK;
}

Status ArmContext::SetNumThreads(int num_threads) {
    num_threads_ = MIN(MAX(num_threads, 1), OMP_CORES_);
    return TNN_OK;
}

int ArmContext::GetNumThreads() {
    return num_threads_;
}

void* ArmContext::GetSharedWorkSpace(size_t size) {
    return GetSharedWorkSpace(size, 0);
}

void* ArmContext::GetSharedWorkSpace(size_t size, int index) {
    while(work_space_.size() < index + 1) {
        work_space_.push_back(RawBuffer(ROUND_UP(size, 64)));
    }
    if (work_space_[index].GetBytesSize() < size) {
        work_space_[index] = RawBuffer(ROUND_UP(size, 64));
    }
    return work_space_[index].force_to<void*>();
}

}  // namespace TNN_NS
