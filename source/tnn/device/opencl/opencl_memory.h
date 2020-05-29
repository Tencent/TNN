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

#ifndef TNN_SOURCE_TNN_DEVICE_OPENCL_OPENCL_MEMORY_H_
#define TNN_SOURCE_TNN_DEVICE_OPENCL_OPENCL_MEMORY_H_
#include "tnn/core/common.h"

namespace TNN_NS {

enum OpenCLMemoryType { TNN_CL_BUFFER = 0, TNN_CL_IMAGE = 1 };

// @brief OpenCLMemory data store in platform and can be shared
class OpenCLMemory {
public:
    // @brief create OpenCLMemory with type
    // @param type: the type of memory
    explicit OpenCLMemory(OpenCLMemoryType type);

    ~OpenCLMemory();

    // @brief get data pointer
    void* GetData() const;

    // @brief set data pointer
    void SetData(void* data_ptr, bool own_data = false);

    // @brief get memory type
    OpenCLMemoryType GetMemoryType() const;

    // @brief set memory type
    void SetMemoryType(OpenCLMemoryType type);

private:
    // remove all assignment operator
    OpenCLMemory(const OpenCLMemory& memory)  = delete;
    OpenCLMemory(const OpenCLMemory&& memory) = delete;
    OpenCLMemory& operator=(const OpenCLMemory&) = delete;
    OpenCLMemory& operator=(const OpenCLMemory&&) = delete;

private:
    // data pointer
    void* data_ = nullptr;
    // memory type
    OpenCLMemoryType mem_type_ = TNN_CL_IMAGE;
    // own_data_ decide whether need to release data
    bool own_data_ = false;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_OPENCL_OPENCL_MEMORY_H_
