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

#include <string.h>
#include <string>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <NvInfer.h>

#include "tnn/network/tensorrt/utils.h"
#include "tnn/core/macro.h"

namespace TNN_NS {

std::string GetGpuType(int gpu_id) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu_id);
    int length = strlen(prop.name);
    for(int i=0;i<length;i++) {
        char c = prop.name[i];
        if (((c >= 'a') && (c<='z')) ||
            ((c >= 'A') && (c<='Z')) ||
            ((c >= '0') && (c<='9'))) {
            continue;
        }
        prop.name[i] = '_';
    }
    return std::string(prop.name);
}

std::string GetGpuArch(int gpu_id) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu_id);
    char ss[50];
    sprintf(ss, "sm%1d%1d", prop.major, prop.minor);
    return std::string(ss);
}

std::string GetCudaVersion() {
    int version_num;

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#else
    version_num = CUDART_VERSION;
#endif 

    char ss[50];
    sprintf(ss, "%02d", version_num / 1000);

    return std::string(ss);
}

std::string GetTrtVersion() {
    int version_num;

#ifndef NV_TENSORRT_MAJOR
#error NV_TENSORRT_MAJOR Undefined!
#else
    version_num = NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR * 10 + NV_TENSORRT_PATCH;
#endif 

    char ss[50];
    sprintf(ss, "%3d", version_num);

    return std::string(ss);
}

DataType ConvertTRTDataType(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT :
            return DATA_TYPE_FLOAT;
        case nvinfer1::DataType::kHALF :
            return DATA_TYPE_HALF;
        default:
            return DATA_TYPE_FLOAT;
    }
}

DataFormat ConvertTRTDataFormat(nvinfer1::TensorFormat format) {
    switch (format) {
        case nvinfer1::TensorFormat::kNCHW :
            return DATA_FORMAT_NCHW;
        case nvinfer1::TensorFormat::kCHW2 :
            return DATA_FORMAT_NC2HW2;
        case nvinfer1::TensorFormat::kCHW4 :
            return DATA_FORMAT_NC4HW4;
        case nvinfer1::TensorFormat::kCHW16 :
            return DATA_FORMAT_NC16HW16;
        default:
            return DATA_FORMAT_NCHW;
    }
}

}  //  namespace TNN_NS
