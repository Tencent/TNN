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

#ifndef TNN_INCLUDE_TNN_CORE_STATUS_H_
#define TNN_INCLUDE_TNN_CORE_STATUS_H_

#include <memory>
#include <string>
#include <vector>

#include "tnn/core/macro.h"

#pragma warning(push)
#pragma warning(disable : 4251)

namespace TNN_NS {

enum StatusCode {

    RPD_OK = 0x0,

    // param errcode
    RPDERR_PARAM_ERR        = 0x1000,
    RPDERR_INVALID_NETCFG   = 0x1002,
    RPDERR_INVALID_LAYERCFG = 0x1003,
    RPDERR_NULL_PARAM       = 0x1004,
    RPDERR_INVALID_GROUP    = 0x1005,
    RPDERR_INVALID_AXIS     = 0x1006,

    // network errcode
    RPDERR_NET_ERR       = 0x2000,
    RPDERR_UNSUPPORT_NET = 0x2001,

    // layer errcode
    RPDERR_LAYER_ERR     = 0x3000,
    RPDERR_UNKNOWN_LAYER = 0x3001,
    RPDERR_CREATE_LAYER  = 0x3002,
    RPDERR_INIT_LAYER    = 0x3003,
    RPDERR_INVALID_DATA  = 0x3004,
    RPDERR_ELT_UNSUP_OP  = 0x3005,

    // model errcode
    RPDERR_MODEL_ERR     = 0x4000,
    RPDERR_INVALID_MODEL = 0x4001,
    RPDERR_FIND_MODEL    = 0x4002,

    // instance errcode
    RPDERR_INST_ERR         = 0x5000,
    RPDERR_MAXINST_COUNT    = 0x5001,
    RPDERR_ALLOC_INSTANCE   = 0x5002,
    RPDERR_INVALID_INSTANCE = 0x5003,
    RPDERR_CONTEXT_ERR      = 0x5004,

    // common errcode
    RPDERR_COMMON_ERROR     = 0x6000,
    RPDERR_OUTOFMEMORY      = 0x6001,
    RPDERR_INVALID_INPUT    = 0x6002,
    RPDERR_FIND_RESOURCE    = 0x6003,
    RPDERR_NO_RESULT        = 0x6004,
    RPDERR_LOAD_MODEL       = 0x6005,
    RPDERR_PACK_MODEL       = 0x6006,
    RPDERR_SET_CPU_AFFINITY = 0x6007,

    // forward memory error
    RPDERR_NOT_SUPPORT_SET_FORWARD_MEM           = 0x8000,
    RPDERR_FORWARD_MEM_NOT_SET                   = 0x8001,
    RPDERR_SHARED_MEMORY_FORWARD_NOT_SAME_THREAD = 0x8003,
    RPDERR_SHARE_MEMORY_MODE_NOT_SUPPORT         = 0x8004,

    // device
    RPDERR_DEVICE_NOT_SUPPORT                 = 0x9000,
    RPDERR_DEVICE_LIBRARY_LOAD                = 0x9001,
    RPDERR_DEVICE_CONTEXT_CREATE              = 0x9002,
    RPDERR_DEVICE_INVALID_COMMAND_QUEUE       = 0x9003,
    RPDERR_DEVICE_ACC_DATA_FORMAT_NOT_SUPPORT = 0x9004,

    // OpenCL
    RPDERR_OPENCL_FINISH_ERROR      = 0xA000,
    RPDERR_OPENCL_API_ERROR         = 0xA001,
    RPDERR_OPENCL_RUNTIME_ERROR     = 0xA002,
    RPDERR_OPENCL_ACC_INIT_ERROR    = 0xA003,
    RPDERR_OPENCL_ACC_RESHAPE_ERROR = 0xA004,
    RPDERR_OPENCL_ACC_FORWARD_ERROR = 0xA005,
    RPDERR_OPENCL_KERNELBUILD_ERROR = 0xA006,
    RPDERR_OPENCL_MEMALLOC_ERROR    = 0xA007,
    RPDERR_OPENCL_MEMMAP_ERROR      = 0xA008,
    RPDERR_OPENCL_MEMUNMAP_ERROR    = 0xA009,

    // SNPE
    RPDERR_SNPE_API_ERROR = 0xB001,

    // Atlas
    RPDERR_ATLAS_RUNTIME_ERROR    = 0xC001,
    RPDERR_ATLAS_TIMEOUT_ERROR    = 0xC002,
    RPDERR_ATLAS_MALLOC_ERROR     = 0xC002,
    RPDERR_ATLAS_GRAPH_INIT_ERROR = 0xC003,

    // Hiai
    RPDERR_HIAI_API_ERROR = 0xD001,

    // Quantize
    RPDERR_QUANTIZE_ERROR = 0xF001,
};

class PUBLIC Status {
public:
    ~Status();
    Status(int code = RPD_OK, std::string message = "OK");

    Status &operator=(int code);

    bool operator==(int code_);
    bool operator!=(int code_);
    operator int();
    operator bool();
    std::string description();

private:
    int code_ = 0;
    std::string message_ = "";
};

}  // namespace TNN_NS

#pragma warning(pop)

#endif  // TNN_INCLUDE_TNN_CORE_STATUS_H_
