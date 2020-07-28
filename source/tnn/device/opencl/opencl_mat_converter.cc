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

#include "tnn/device/opencl/opencl_mat_converter.h"

#include "tnn/device/opencl/opencl_utils.h"

namespace TNN_NS {

Status OpenCLMatConverterAcc::Resize(Mat& src, Mat& dst, ResizeParam param, void* command_queue = NULL) {
    Status ret            = TNN_OK;
    auto cl_command_queue = static_cast<cl::CommandQueue *>(command_queue);
    if (cl_command_queue == nullptr) {
        LOGE("Get OpenCL command queue failed!\n");
        return Status(TNNERR_NULL_PARAM, "Get OpenCL command queue failed!");
    }
    const std::string key = "Resize";
    OpenCLExecuteUnit unit;
    if(execute_map_.count(key) == 0) {
        std::string program_name = "normalize";
        std::string kernel_name = "Bilinear";
        ret = CreateExecuteUnit(unit, program_name, kernel_name);
        if(ret != TNN_OK) {
            return ret;
        }
        execute_map_[key] = unit; 
    }

    
    
}

Status OpenCLMatConverterAcc::Crop(Mat& src, Mat& dst, CropParam param, void* command_queue = NULL) {
    Status ret            = TNN_OK;
    auto cl_command_queue = static_cast<cl::CommandQueue *>(command_queue);
    if (cl_command_queue == nullptr) {
        LOGE("Get OpenCL command queue failed!\n");
        return Status(TNNERR_NULL_PARAM, "Get OpenCL command queue failed!");
    }
    const std::string key = "Crop"; 
    if(execute_map_.count(key) == 0) {
        std::string program_name = "copy";
        std::string kernel_name = "CopyImage";
        ret = CreateExecuteUnit(unit, program_name, kernel_name);
        if(ret != TNN_OK) {
            return ret;
        }
        execute_map_[key] = unit; 
    }
}

Status OpenCLMatConverterAcc::WarpAffine(Mat& src, Mat& dst, WarpAffineParam param, void* command_queue = NULL) {
    Status ret            = TNN_OK;
    auto cl_command_queue = static_cast<cl::CommandQueue *>(command_queue);
    if (cl_command_queue == nullptr) {
        LOGE("Get OpenCL command queue failed!\n");
        return Status(TNNERR_NULL_PARAM, "Get OpenCL command queue failed!");
    } 
    if(execute_map_.count("WarpAffine") == 0) {        

    }
}

}  // namespace TNN_NS
