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

#include "tnn/device/opencl/imagebuffer_convertor.h"

namespace TNN_NS {

Status ImageBufferConvertor::ConvertImageToBuffer(const OpenCLMemory *image, const OpenCLBufferFormat type,
                                                  DimsVector dims, OpenCLMemory *buffer, bool need_wait) {
    LOGD("start ConvertImageToBuffer !\n");
    auto image_shape = GetImageShape(image);

    std::string kernel_name;
    if (type == NHWC_BUFFER) {
        kernel_name = "ImageToNHWCBuffer";
    } else if (type == NCHW_BUFFER) {
        kernel_name = "ImageToNCHWBuffer";
    } else if (type == CONV2D_FILTER) {
        kernel_name = "Conv2DFilterImageToBuffer";
    } else if (type == ARGUMENT) {
        kernel_name = "ArgImageToBuffer";
    } else {
        LOGE("not support such type !!! \n");
        return Status(TNNERR_OPENCL_API_ERROR, "type not support");
    }

    Status ret = TNN_OK;
    if (image_to_buffer_unit_.ocl_kernel.get() == nullptr || image_to_buffer_kernelname_ != kernel_name) {
        image_to_buffer_kernelname_ = kernel_name;
        std::set<std::string> build_options;

        ret = CreateExecuteUnit(image_to_buffer_unit_, "image_to_buffer", kernel_name, build_options);
        CHECK_TNN_OK(ret)
    }

    image_to_buffer_unit_.global_work_size = {static_cast<uint32_t>(image_shape[0]),
                                            static_cast<uint32_t>(image_shape[1])};

    uint32_t idx = 0;
    image_to_buffer_unit_.ocl_kernel.setArg(idx++, image_to_buffer_unit_.global_work_size[0]);
    image_to_buffer_unit_.ocl_kernel.setArg(idx++, image_to_buffer_unit_.global_work_size[1]);
    image_to_buffer_unit_.ocl_kernel.setArg(idx++, GetOpenCLBuffer(buffer));

    if (type == CONV2D_FILTER) {
        //channel * height * width
        const int ic_w_h_size = dims[1] * dims[2] * dims[3];
        //height * width
        const int w_h_size    = dims[2] * dims[3];
        int kernel_shape[2]   = {dims[2], dims[3]};
        //batch
        image_to_buffer_unit_.ocl_kernel.setArg(idx++, static_cast<uint32_t>(dims[0]));
        image_to_buffer_unit_.ocl_kernel.setArg(idx++, sizeof(kernel_shape), kernel_shape);
        image_to_buffer_unit_.ocl_kernel.setArg(idx++, static_cast<uint32_t>(ic_w_h_size));
        image_to_buffer_unit_.ocl_kernel.setArg(idx++, static_cast<uint32_t>(w_h_size));
    } else if (type == ARGUMENT) {
        //batch
        image_to_buffer_unit_.ocl_kernel.setArg(idx++, static_cast<uint32_t>(dims[0]));
    } else {
        //height
        image_to_buffer_unit_.ocl_kernel.setArg(idx++, static_cast<uint32_t>(dims[2]));
        //width
        image_to_buffer_unit_.ocl_kernel.setArg(idx++, static_cast<uint32_t>(dims[3]));
        //channel
        image_to_buffer_unit_.ocl_kernel.setArg(idx++, static_cast<uint32_t>(dims[1]));
    }

    image_to_buffer_unit_.ocl_kernel.setArg(idx++, GetOpenCLImage(image));

    image_to_buffer_unit_.local_work_size = LocalWS2DDefault(image_to_buffer_unit_);

    ret = RunKernel(image_to_buffer_unit_.ocl_kernel, image_to_buffer_unit_.global_work_size,
                    image_to_buffer_unit_.local_work_size, opencl_command_queue_, "ConvertImageToBuffer");
    CHECK_TNN_OK(ret)

    if (need_wait) {
        //sync
        opencl_command_queue_->finish();
    }
    LOGD("end convertImageToBuffer !\n");
    return TNN_OK;
}

Status ImageBufferConvertor::ConvertBufferToImage(const OpenCLMemory *buffer, const OpenCLBufferFormat type,
                                                  DimsVector dims, OpenCLMemory *image, bool need_wait) {
    LOGD("start ConvertBufferToImage !\n");
    auto image_shape = GetImageShape(image);

    std::string kernel_name;
    if (type == CONV2D_FILTER) {
        kernel_name = "Conv2DFilterBufferToImage";
    } else if (type == DW_CONV2D_FILTER) {
        kernel_name = "DWFilterBufferToImage";
    } else if (type == NHWC_BUFFER) {
        kernel_name = "NHWCBufferToImage";
    } else if (type == NCHW_BUFFER) {
        kernel_name = "NCHWBufferToImage";
    } else if (type == ARGUMENT) {
        kernel_name = "ArgBufferToImage";
    } else {
        LOGE("not support such type !!! \n");
        return Status(TNNERR_OPENCL_API_ERROR, "type not support");
    }

    Status ret = TNN_OK;
    if (buffer_to_image_unit_.ocl_kernel.get() == nullptr || buffer_to_image_kernelname_ != kernel_name) {
        buffer_to_image_kernelname_ = kernel_name;
        std::set<std::string> build_options;

        ret = CreateExecuteUnit(buffer_to_image_unit_, "buffer_to_image", kernel_name, build_options);
        CHECK_TNN_OK(ret)
    }

    buffer_to_image_unit_.global_work_size = {static_cast<uint32_t>(image_shape[0]),
                                            static_cast<uint32_t>(image_shape[1])};

    uint32_t idx = 0;
    buffer_to_image_unit_.ocl_kernel.setArg(idx++, buffer_to_image_unit_.global_work_size[0]);
    buffer_to_image_unit_.ocl_kernel.setArg(idx++, buffer_to_image_unit_.global_work_size[1]);
    buffer_to_image_unit_.ocl_kernel.setArg(idx++, GetOpenCLBuffer(buffer));

    if (type == CONV2D_FILTER) {
        //channel * height * width
        const int ic_w_h_size = dims[1] * dims[2] * dims[3];
        //height * width
        const int w_h_size    = dims[2] * dims[3];
        int kernel_shape[2]   = {dims[2], dims[3]};
        buffer_to_image_unit_.ocl_kernel.setArg(idx++, static_cast<uint32_t>(dims[0]));
        buffer_to_image_unit_.ocl_kernel.setArg(idx++, sizeof(kernel_shape), kernel_shape);
        buffer_to_image_unit_.ocl_kernel.setArg(idx++, static_cast<uint32_t>(ic_w_h_size));
        buffer_to_image_unit_.ocl_kernel.setArg(idx++, static_cast<uint32_t>(w_h_size));
    } else if (type == DW_CONV2D_FILTER) {
        //height * width
        const int w_h_size  = dims[2] * dims[3];
        int kernel_shape[4] = {dims[0], dims[1], dims[2], dims[3]};
        buffer_to_image_unit_.ocl_kernel.setArg(idx++, sizeof(kernel_shape), kernel_shape);
        buffer_to_image_unit_.ocl_kernel.setArg(idx++, static_cast<uint32_t>(w_h_size));
    } else if (type == ARGUMENT) {
        //batch
        buffer_to_image_unit_.ocl_kernel.setArg(idx++, static_cast<uint32_t>(dims[0]));
    } else {
        //height
        buffer_to_image_unit_.ocl_kernel.setArg(idx++, static_cast<uint32_t>(dims[2]));
        //width
        buffer_to_image_unit_.ocl_kernel.setArg(idx++, static_cast<uint32_t>(dims[3]));
        //channel
        buffer_to_image_unit_.ocl_kernel.setArg(idx++, static_cast<uint32_t>(dims[1]));
    }

    buffer_to_image_unit_.ocl_kernel.setArg(idx++, GetOpenCLImage(image));

    buffer_to_image_unit_.local_work_size = LocalWS2DDefault(buffer_to_image_unit_);

    ret = RunKernel(buffer_to_image_unit_.ocl_kernel, buffer_to_image_unit_.global_work_size,
                    buffer_to_image_unit_.local_work_size, opencl_command_queue_, "ConvertBufferToImage");
    CHECK_TNN_OK(ret)

    if (need_wait) {
        opencl_command_queue_->finish();
    }
    LOGD("end ConvertBufferToImage !\n");
    return TNN_OK;
}

Status ImageBufferConvertor::ConvertBufferToBuffer(const OpenCLMemory *input, const OpenCLBufferFormat type,
                                                   DimsVector dims, OpenCLMemory *output, bool need_wait) {
    LOGD("start ConvertBufferToBuffer !\n");

    std::string kernel_name;
    if (type == CONV2D_FILTER) {
        kernel_name = "Conv2DFilterBufferToBuffer";
    } else if (type == DW_CONV2D_FILTER) {
        kernel_name = "DWFilterBufferToBuffer";
    } else if (type == ARGUMENT && dims.size() == 1) {
        kernel_name = "ArgBufferToBuffer";
    } else {
        LOGE("not support such type !!! \n");
        return Status(TNNERR_OPENCL_API_ERROR, "type not support");
    }

    Status ret = TNN_OK;
    if (buffer_to_buffer_unit_.ocl_kernel.get() == nullptr || buffer_to_buffer_kernelname_ != kernel_name) {
        buffer_to_buffer_kernelname_ = kernel_name;
        std::set<std::string> build_options;

        ret = CreateExecuteUnit(buffer_to_buffer_unit_, "buffer_to_buffer", kernel_name, build_options);
        CHECK_TNN_OK(ret)
    }

    if (type == CONV2D_FILTER) {
        buffer_to_buffer_unit_.global_work_size.push_back(ROUND_UP(dims[0], 4));
        buffer_to_buffer_unit_.global_work_size.push_back(dims[2] * dims[3] * ROUND_UP(dims[1], 4));
    } else if (type == DW_CONV2D_FILTER) {
        buffer_to_buffer_unit_.global_work_size.push_back(dims[2] * dims[3]);
        buffer_to_buffer_unit_.global_work_size.push_back(UP_DIV(dims[1], 4));
    } else if (type == ARGUMENT && dims.size() == 1) {
        buffer_to_buffer_unit_.global_work_size.push_back(UP_DIV(dims[0], 4));
        buffer_to_buffer_unit_.global_work_size.push_back(1);
    } else {
        LOGE("not support such type !!! \n");
        return Status(TNNERR_OPENCL_API_ERROR, "type not support");
    }

    uint32_t idx = 0;
    buffer_to_buffer_unit_.ocl_kernel.setArg(idx++, buffer_to_buffer_unit_.global_work_size[0]);
    buffer_to_buffer_unit_.ocl_kernel.setArg(idx++, buffer_to_buffer_unit_.global_work_size[1]);
    buffer_to_buffer_unit_.ocl_kernel.setArg(idx++, GetOpenCLBuffer(input));

    if (type == CONV2D_FILTER) {
        //height * width
        const int w_h_size  = dims[2] * dims[3];
        int kernel_shape[2] = {dims[2], dims[3]};
        //batch
        buffer_to_buffer_unit_.ocl_kernel.setArg(idx++, static_cast<uint32_t>(dims[0]));
        //channel
        buffer_to_buffer_unit_.ocl_kernel.setArg(idx++, static_cast<uint32_t>(dims[1]));
        buffer_to_buffer_unit_.ocl_kernel.setArg(idx++, sizeof(kernel_shape), kernel_shape);
        buffer_to_buffer_unit_.ocl_kernel.setArg(idx++, static_cast<uint32_t>(w_h_size));
    } else if (type == DW_CONV2D_FILTER) {
        //height * width
        const int w_h_size  = dims[2] * dims[3];
        int kernel_shape[4] = {dims[0], dims[1], dims[2], dims[3]};
        buffer_to_buffer_unit_.ocl_kernel.setArg(idx++, sizeof(kernel_shape), kernel_shape);
        buffer_to_buffer_unit_.ocl_kernel.setArg(idx++, static_cast<uint32_t>(w_h_size));
    } else if (type == ARGUMENT) {
        //batch
        buffer_to_buffer_unit_.ocl_kernel.setArg(idx++, static_cast<uint32_t>(dims[0]));
    } else {
        LOGE("not support such type !!! \n");
        return Status(TNNERR_OPENCL_API_ERROR, "type not support");
    }

    buffer_to_buffer_unit_.ocl_kernel.setArg(idx++, GetOpenCLBuffer(output));

    buffer_to_buffer_unit_.local_work_size = LocalWS2DDefault(buffer_to_buffer_unit_);

    ret = RunKernel(buffer_to_buffer_unit_.ocl_kernel, buffer_to_buffer_unit_.global_work_size,
                    buffer_to_buffer_unit_.local_work_size, opencl_command_queue_, "ConvertBufferToBuffer");
    CHECK_TNN_OK(ret)

    if (need_wait) {
        opencl_command_queue_->finish();
    }
    LOGD("end ConvertBufferToBuffer !\n");
    return TNN_OK;
}

}  // namespace TNN_NS
