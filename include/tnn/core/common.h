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

#ifndef TNN_INCLUDE_TNN_CORE_COMMON_H_
#define TNN_INCLUDE_TNN_CORE_COMMON_H_

#include <functional>
#include <string>
#include <vector>

#include "tnn/core/macro.h"

#pragma warning(push)
#pragma warning(disable : 4251)

namespace TNN_NS {

typedef std::function<void(void)> Callback;

typedef enum {
    DATA_TYPE_FLOAT = 0,
    DATA_TYPE_HALF  = 1,
    DATA_TYPE_INT8  = 2,
    DATA_TYPE_INT32 = 3,
    DATA_TYPE_BFP16 = 4
} DataType;

typedef enum {
    // decided by device
    DATA_FORMAT_AUTO   = -1,
    DATA_FORMAT_NCHW   = 0,
    DATA_FORMAT_NHWC   = 1,
    DATA_FORMAT_NHWC4  = 2,
    DATA_FORMAT_NC4HW4 = 3,
    DATA_FORMAT_NCDHW  = 4,
    DATA_FORMAT_NHC4W4 = 5,
    DATA_FORMAT_NC8HW8 = 6,
} DataFormat;

typedef enum {
    // auto precision, each device choose default precision.
    // ARM: prefer fp16
    // OPENCL: prefer fp16
    // METAL: prefer fp16
    PRECISION_AUTO = -1,
    // Normal precision
    // ARM: run fp16 if device support fp16, else run fp32.
    // OPNECL: run with mixed pricision
    // METAL: run with fp16
    PRECISION_NORMAL = 0,
    // High precision
    // ARM: run with fp32
    // OPENCL: run with fp32
    // METAL: run with fp16
    PRECISION_HIGH = 1,
    // Low precision
    // ARM: run with bfp16
    // OPENCL: run with fp16
    // METAL: run with fp16
    PRECISION_LOW = 2
} Precision;

typedef enum {
    NETWORK_TYPE_DEFAULT    = 0,
    NETWORK_TYPE_OPENVINO   = 0x1000,
    NETWORK_TYPE_COREML     = 0x2000,
    NETWORK_TYPE_SNPE       = 0x3000,
    NETWORK_TYPE_HIAI       = 0x4000,
    NETWORK_TYPE_ATLAS      = 0x5000,
    NETWORK_TYPE_HUAWEI_NPU = 0x6000,
    NETWORK_TYPE_RK_NPU     = 0x7000
} NetworkType;

typedef enum {
    DEVICE_NAIVE      = 0x0000,
    DEVICE_X86        = 0x0010,
    DEVICE_ARM        = 0x0020,
    DEVICE_OPENCL     = 0x1000,
    DEVICE_METAL      = 0x1010,
    DEVICE_CUDA       = 0x1020,
    DEVICE_DSP        = 0x1030,
    DEVICE_ATLAS      = 0x1040,
    DEVICE_HUAWEI_NPU = 0x1050,
    DEVICE_RK_NPU     = 0x1060,
} DeviceType;

typedef enum {
    // default
    SHARE_MEMORY_MODE_DEFAULT = 0,
    // same thread tnn instance share blob memory
    SHARE_MEMORY_MODE_SHARE_ONE_THREAD = 1,
    // set blob memory from external, different thread share blob memory need
    // synchronize
    SHARE_MEMORY_MODE_SET_FROM_EXTERNAL = 2
} ShareMemoryMode;

typedef enum {
    MODEL_TYPE_TNN      = 0x0001,
    MODEL_TYPE_NCNN     = 0x0100,
    MODEL_TYPE_OPENVINO = 0x1000,
    MODEL_TYPE_COREML   = 0x2000,
    MODEL_TYPE_SNPE     = 0x3000,
    MODEL_TYPE_HIAI     = 0x4000,
    MODEL_TYPE_ATLAS    = 0x5000
} ModelType;

using DimsVector = std::vector<int>;

//@brief Config used to create tnn instance, config
// device type, network type and share memory mode.
struct PUBLIC NetworkConfig {
    // device type default cpu
    DeviceType device_type = DEVICE_ARM;

    // device id default 0
    int device_id = 0;

    // blob data format decided by device
    DataFormat data_format = DATA_FORMAT_AUTO;

    // network type default internal
    NetworkType network_type = NETWORK_TYPE_DEFAULT;

    // raidnet instances not share memory with others
    ShareMemoryMode share_memory_mode = SHARE_MEMORY_MODE_DEFAULT;

    // dependent library path
    std::vector<std::string> library_path = {};

    // compute precision
    Precision precision = PRECISION_AUTO;

    // cache path to store possible cache models or opt kernel
    std::string cache_path = "";

    // network init or reshape may cost more time to select opt kernel implement if enable tune kernel
    // cache_path can set to store tune kernel info.
    bool enable_tune_kernel = false;
};

struct PUBLIC ModelConfig {
    ModelType model_type = MODEL_TYPE_TNN;

    // tnn model need two params: order is proto content, model content.
    // ncnn need two: params: order is param, weights.
    // openvino model need two params: order is xml content, model path.
    // coreml model need one param: coreml model dir.
    // snpe model need one param: dlc model dir.
    // hiai model need two params: order is model name, model_file_path.
    // atlas model need one param: config string.
    std::vector<std::string> params = {};
};

}  // namespace TNN_NS

#pragma warning(pop)

#endif  // TNN_INCLUDE_TNN_CORE_COMMON_H_
