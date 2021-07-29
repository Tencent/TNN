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

    TNN_OK = 0x0,

    // param errcode
    TNNERR_PARAM_ERR        = 0x1000,
    TNNERR_INVALID_NETCFG   = 0x1002,
    TNNERR_INVALID_LAYERCFG = 0x1003,
    TNNERR_NULL_PARAM       = 0x1004,
    TNNERR_INVALID_GROUP    = 0x1005,
    TNNERR_INVALID_AXIS     = 0x1006,

    // network errcode
    TNNERR_NET_ERR       = 0x2000,
    TNNERR_UNSUPPORT_NET = 0x2001,

    // layer errcode
    TNNERR_LAYER_ERR     = 0x3000,
    TNNERR_UNKNOWN_LAYER = 0x3001,
    TNNERR_CREATE_LAYER  = 0x3002,
    TNNERR_INIT_LAYER    = 0x3003,
    TNNERR_INVALID_DATA  = 0x3004,
    TNNERR_ELT_UNSUP_OP  = 0x3005,

    // model errcode
    TNNERR_MODEL_ERR     = 0x4000,
    TNNERR_INVALID_MODEL = 0x4001,
    TNNERR_FIND_MODEL    = 0x4002,

    // instance errcode
    TNNERR_INST_ERR         = 0x5000,
    TNNERR_MAXINST_COUNT    = 0x5001,
    TNNERR_ALLOC_INSTANCE   = 0x5002,
    TNNERR_INVALID_INSTANCE = 0x5003,
    TNNERR_CONTEXT_ERR      = 0x5004,

    // common errcode
    TNNERR_COMMON_ERROR     = 0x6000,
    TNNERR_OUTOFMEMORY      = 0x6001,
    TNNERR_INVALID_INPUT    = 0x6002,
    TNNERR_FIND_RESOURCE    = 0x6003,
    TNNERR_NO_RESULT        = 0x6004,
    TNNERR_LOAD_MODEL       = 0x6005,
    TNNERR_PACK_MODEL       = 0x6006,
    TNNERR_SET_CPU_AFFINITY = 0x6007,
    TNNERR_OPEN_FILE        = 0x6008,

    // forward memory error
    TNNERR_NOT_SUPPORT_SET_FORWARD_MEM           = 0x8000,
    TNNERR_FORWARD_MEM_NOT_SET                   = 0x8001,
    TNNERR_SHARED_MEMORY_FORWARD_NOT_SAME_THREAD = 0x8003,
    TNNERR_SHARE_MEMORY_MODE_NOT_SUPPORT         = 0x8004,

    // device
    TNNERR_DEVICE_NOT_SUPPORT                 = 0x9000,
    TNNERR_DEVICE_LIBRARY_LOAD                = 0x9001,
    TNNERR_DEVICE_CONTEXT_CREATE              = 0x9002,
    TNNERR_DEVICE_INVALID_COMMAND_QUEUE       = 0x9003,
    TNNERR_DEVICE_ACC_DATA_FORMAT_NOT_SUPPORT = 0x9004,

    // OpenCL
    TNNERR_OPENCL_FINISH_ERROR      = 0xA000,
    TNNERR_OPENCL_API_ERROR         = 0xA001,
    TNNERR_OPENCL_RUNTIME_ERROR     = 0xA002,
    TNNERR_OPENCL_ACC_INIT_ERROR    = 0xA003,
    TNNERR_OPENCL_ACC_RESHAPE_ERROR = 0xA004,
    TNNERR_OPENCL_ACC_FORWARD_ERROR = 0xA005,
    TNNERR_OPENCL_KERNELBUILD_ERROR = 0xA006,
    TNNERR_OPENCL_MEMALLOC_ERROR    = 0xA007,
    TNNERR_OPENCL_MEMMAP_ERROR      = 0xA008,
    TNNERR_OPENCL_MEMUNMAP_ERROR    = 0xA009,
    TNNERR_OPENCL_UNSUPPORT_ERROR   = 0xA00A,

    // SNPE
    TNNERR_SNPE_API_ERROR = 0xB001,

    // Atlas
    TNNERR_ATLAS_RUNTIME_ERROR    = 0xC001,
    TNNERR_ATLAS_TIMEOUT_ERROR    = 0xC002,
    TNNERR_ATLAS_MALLOC_ERROR     = 0xC003,
    TNNERR_ATLAS_GRAPH_INIT_ERROR = 0xC004,

    // Hiai
    TNNERR_HIAI_API_ERROR = 0xD001,

    // Huawei NPU
    TNNERR_NPU_LOAD_ERROR      = 0xE000,
    TNNERR_NPU_UNSUPPORT_ERROR = 0xE001,
    TNNERR_NPU_HIAI_API_ERROR  = 0xE002,

    // Cuda
    TNNERR_CUDA_TENSORRT_ERROR = 0xF001,
    TNNERR_CUDA_SYNC_ERROR     = 0xF002,
    TNNERR_CUDA_MEMCPY_ERROR   = 0xF003,

    // TNN CONVERT
    TNN_CONVERT_OK                 = 0x10000,
    TNNERR_CONVERT_UNSUPPORT_LAYER = 0x10001,
    TNNERR_CONVERT_GENERATE_MODEL  = 0x10002,
    TNNERR_CONVERT_INVALID_MODEL   = 0x10003,
    TNNERR_CONVERT_UNSUPPORT_PASS  = 0x10004,
    TNNERR_CONVERT_OPTIMIZE_ERROR  = 0x10005,

    // Quantize
    TNNERR_QUANTIZE_ERROR = 0x20001,
};

class PUBLIC Status {
public:
    ~Status();
    Status(int code = TNN_OK, std::string message = "OK");

    Status &operator=(int code);

    bool operator==(int code_);
    bool operator!=(int code_);
    operator int();
    operator bool();
    std::string description();

private:
    int code_            = 0;
    std::string message_ = "";
};

}  // namespace TNN_NS

#pragma warning(pop)

#endif  // TNN_INCLUDE_TNN_CORE_STATUS_H_
