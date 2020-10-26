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

#ifndef TNN_SOURCE_TNN_DEVICE_OPENCL_OPENCL_EXECUTE_UNIT_H_
#define TNN_SOURCE_TNN_DEVICE_OPENCL_OPENCL_EXECUTE_UNIT_H_

#include "tnn/device/opencl/opencl_runtime.h"

struct OpenCLExecuteUnit {
    cl::Kernel ocl_kernel;
    uint32_t workgroupsize_max;
    std::vector<uint32_t> global_work_size = {};
    std::vector<uint32_t> local_work_size = {};
    uint32_t sub_group_size;
    uint64_t local_mem_size;
};

#endif  // TNN_SOURCE_TNN_DEVICE_OPENCL_OPENCL_EXECUTE_UNIT_H_
