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

#ifndef TNN_SOURCE_TNN_DEVICE_OPENCL_IMAGEBUFFER_CONVERTOR_H_
#define TNN_SOURCE_TNN_DEVICE_OPENCL_IMAGEBUFFER_CONVERTOR_H_

#include "tnn/core/macro.h"
#include "tnn/device/opencl/opencl_context.h"
#include "tnn/device/opencl/opencl_runtime.h"
#include "tnn/device/opencl/opencl_utils.h"

namespace TNN_NS {

class ImageBufferConvertor {
public:
    ImageBufferConvertor(OpenCLRuntime *opencl_runtime, cl::CommandQueue *opencl_command_queue)
        : opencl_runtime_(opencl_runtime), opencl_command_queue_(opencl_command_queue) {}
    Status ConvertImageToBuffer(const OpenCLMemory *input, const OpenCLBufferFormat type, DimsVector dims,
                                OpenCLMemory *output, bool need_wait = false);
    Status ConvertBufferToImage(const OpenCLMemory *input, const OpenCLBufferFormat type, DimsVector dims,
                                OpenCLMemory *output, bool need_wait = false);
    Status ConvertBufferToBuffer(const OpenCLMemory *input, const OpenCLBufferFormat type, DimsVector dims,
                                 OpenCLMemory *output, bool need_wait = false);

private:
    OpenCLRuntime *opencl_runtime_;
    cl::CommandQueue *opencl_command_queue_;
    std::string image_to_buffer_kernelname_ = "";
    OpenCLExecuteUnit image_to_buffer_unit_;
    std::string buffer_to_image_kernelname_ = "";
    OpenCLExecuteUnit buffer_to_image_unit_;
    std::string buffer_to_buffer_kernelname_ = "";
    OpenCLExecuteUnit buffer_to_buffer_unit_;
};

}  // namespace TNN_NS
#endif  // TNN_SOURCE_TNN_DEVICE_OPENCL_IMAGEBUFFER_CONVERTOR_H_
