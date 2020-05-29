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

#include "tnn/device/opencl/opencl_memory.h"
#include "tnn/device/opencl/opencl_runtime.h"

namespace TNN_NS {

OpenCLMemory::OpenCLMemory(OpenCLMemoryType type) {
    mem_type_ = type;
}

OpenCLMemory::~OpenCLMemory() {
    if (own_data_ && data_ != nullptr) {
        if (mem_type_ == TNN_CL_BUFFER) {
            cl::Buffer *ptr = (cl::Buffer *)data_;
            delete ptr;
            ptr = nullptr;
        } else if (mem_type_ == TNN_CL_IMAGE) {
            cl::Image2D *ptr = (cl::Image2D *)data_;
            delete ptr;
            ptr = nullptr;
        }
    }
}

void *OpenCLMemory::GetData() const {
    return data_;
}

void OpenCLMemory::SetData(void *data_ptr, bool own_data) {
    data_     = data_ptr;
    own_data_ = own_data;
}

OpenCLMemoryType OpenCLMemory::GetMemoryType() const {
    return mem_type_;
}

void OpenCLMemory::SetMemoryType(OpenCLMemoryType type) {
    mem_type_ = type;
}

}  // namespace TNN_NS
