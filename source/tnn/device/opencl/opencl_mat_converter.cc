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
#include "tnn/core/macro.h"
#include "tnn/device/opencl/opencl_utils.h"
#include "tnn/memory_manager/blob_memory_size_info.h"
#include "tnn/utils/blob_memory_size_utils.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/string_utils_inner.h"

namespace TNN_NS {

Status OpenCLMatConverterAcc::SetExecuteUnit(
        OpenCLExecuteUnit& unit, Mat& src, Mat& dst,
        const bool copy_flag, const std::string& mat_key) {
    Status ret = TNN_OK;
    if (copy_flag) {
        if (execute_map_.count(mat_key) == 0) {
            std::string program_name = "convert_to_mat";
            std::string kernel_name = "";
            if (N8UC4 == dst.GetMatType()) {
                kernel_name = "CopyToN8UC4";
            } else if (N8UC3 == dst.GetMatType()) {
                kernel_name = "CopyToN8UC3";
            } else if (NGRAY == dst.GetMatType()) {
                program_name = "buffer_to_buffer";
                kernel_name = "BufferToBuffer";
            }
            if (!kernel_name.empty()) {
                ret = CreateExecuteUnit(unit, program_name, kernel_name);
                if (ret != TNN_OK) {
                    return ret;
                }
                execute_map_[mat_key] = unit;
            }
        }
    } else {
        if (execute_map_.count(mat_key) == 0) {
            std::string program_name = "convert_from_mat";
            std::string kernel_name = "";
            if (N8UC4 == src.GetMatType()) {
                kernel_name = "CopyFromN8UC4";
            } else if (N8UC3 == src.GetMatType()) {
                kernel_name = "CopyFromN8UC3";
            } else if (NGRAY == dst.GetMatType()) {
                program_name = "buffer_to_buffer";
                kernel_name = "BufferToBuffer";
            }
            if (!kernel_name.empty()) {
                ret = CreateExecuteUnit(unit, program_name, kernel_name);
                if (ret != TNN_OK) {
                    return ret;
                }
                execute_map_[mat_key] = unit;
            }
        }
    }

    return TNN_OK;
}

Status OpenCLMatConverterAcc::Copy(Mat& src, Mat& dst, void* command_queue) {
    Status ret           = TNN_OK;
    // force float to get the max memeory
    bool copy_flag = false;
    if (src.GetDeviceType() != DEVICE_OPENCL) {//CPU -> GPU
        copy_flag = false;
    } else if (dst.GetDeviceType() != DEVICE_OPENCL){//GPU->CPU
        copy_flag = true;
    }
    // buffer_reset
    BlobMemorySizeInfo info;
    if (src.GetMatType() != NGRAY) {
        info.data_type = DATA_TYPE_FLOAT;
        int batch, channel, height, width;
        batch            = src.GetBatch();
        channel          = src.GetChannel();
        height           = src.GetHeight();
        width            = src.GetWidth();
        //nchw->nhwc
        int image_width  = UP_DIV(channel, 4) * width;
        int image_height = batch * height;
        info.dims.push_back(image_width);
        info.dims.push_back(image_height);
    } else {
        info.data_type = DATA_TYPE_INT8;
        info.dims.push_back(DimsVectorUtils::Count(src.GetDims()));
    }

    auto opencl_runtime   = OpenCLRuntime::GetInstance();
    buffer_size_          = GetBlobMemoryBytesSize(info);
    cl_int ret_cl            = CL_SUCCESS;
    cl::Buffer* cl_buffer = new cl::Buffer(*opencl_runtime->Context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                            (cl::size_type)buffer_size_, nullptr, &ret_cl);
    if (ret_cl != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret_cl)
        if (nullptr != cl_buffer)
            delete cl_buffer;
    } else {
        buffer_.reset(cl_buffer);
    }

    auto cl_command_queue = static_cast<cl::CommandQueue *>(command_queue);
    MatType src_mat_type = src.GetMatType();
    MatType dst_mat_type = dst.GetMatType();

    if (cl_command_queue == nullptr) {
        LOGE("Get OpenCL command queue failed!\n");
        return Status(TNNERR_NULL_PARAM, "Get OpenCL command queue failed!");
    }

    if (nullptr == buffer_) {
        LOGE("OpenCLBlobConverter buffer allocate failed\n");
        return Status(TNNERR_NULL_PARAM, "OpenCLBlobConverter buffer allocate failed!");
    }

    if(src_mat_type != dst_mat_type){
        return Status(TNNERR_PARAM_ERR, "convert type not support yet");
    }

    //create identifier
    std::string mat_key = ToString(src.GetDeviceType()) + "_" + ToString(dst.GetDeviceType());
    //create convert unit only once for every key
    OpenCLExecuteUnit unit;
    ret = SetExecuteUnit(unit, src, dst, copy_flag, mat_key);
    if (ret != TNN_OK) {
        return ret;
    }

    // set copy_arguments
    ret                    = SetConvertArgs(unit, src, dst, false);
    if (ret != TNN_OK) {
        return ret;
    }

    //if src device is cpu, need copy src_mat data to buffer and bind buffer to dst_mat data
    if (src.GetDeviceType() != DEVICE_OPENCL) {
        ret = CopyMatToBufferData(src, cl_command_queue);
        if (ret != TNN_OK) {
            return ret;
        }
        ret = RunConvertUnit(unit, cl_command_queue, false);
        if (ret != TNN_OK) {
            return ret;
        }
    } else {
        ret = RunConvertUnit(unit, cl_command_queue, false);
        if (ret != TNN_OK) {
            return ret;
        }
        ret = CopyBufferDataToMat(dst, cl_command_queue);
        if (ret != TNN_OK) {
            return ret;
        }
    }
    return ret;
}

//enqueueMapBuffer get cpu buffer pointer, copy buffer pointer to mat, enqueueUnmapMemObject.
Status OpenCLMatConverterAcc::CopyBufferDataToMat(Mat &mat, cl::CommandQueue *command_queue) {
    MatType mat_type   = mat.GetMatType();
    DimsVector dims    = mat.GetDims();

    Status ret = CopyBufferToMat(mat, *buffer_, dims, buffer_size_, mat_type, command_queue);
    if (ret != TNN_OK) {
        return ret;
    }

    return TNN_OK;
}

//enqueueMapBuffer get cpu buffer pointer, copy mat to buffer pointer, enqueueUnmapMemObject.
Status OpenCLMatConverterAcc::CopyMatToBufferData(Mat &mat, cl::CommandQueue *command_queue) {
    MatType mat_type   = mat.GetMatType();
    DimsVector dims    = mat.GetDims();

    Status ret = CopyMatToBuffer(mat, *buffer_, dims, buffer_size_, mat_type, command_queue);
    if (ret != TNN_OK) {
        return ret;
    }

    return TNN_OK;
}

Status OpenCLMatConverterAcc::SetConvertArgs(OpenCLExecuteUnit &unit, Mat &src, Mat &dst,
                                              bool convert_to_mat) {
    MatType mat_type = src.GetMatType();
    auto dims        = dst.GetDims();

    cl_int cl_ret;
    // gray mat copy buffer to buffer
    if (NGRAY == mat_type) {
        uint32_t idx = SetExecuteUnit1DSizeInfoDefault(unit, dims);
        if (src.GetDeviceType() != DEVICE_OPENCL) {
            cl::Buffer *dst_buffer = static_cast<cl::Buffer *>(dst.GetData());
            cl_ret = unit.ocl_kernel.setArg(idx++, *buffer_);
            CHECK_CL_SUCCESS(cl_ret);
            cl_ret = unit.ocl_kernel.setArg(idx++, *dst_buffer);
            CHECK_CL_SUCCESS(cl_ret);
            // src_offset
            cl_ret = unit.ocl_kernel.setArg(idx++, 0);
            CHECK_CL_SUCCESS(cl_ret);
            // dst_offset
            cl_ret = unit.ocl_kernel.setArg(idx++, 0);
            CHECK_CL_SUCCESS(cl_ret);
        } else {
            cl::Buffer *src_buffer = static_cast<cl::Buffer *>(src.GetData());
            cl_ret = unit.ocl_kernel.setArg(idx++, *src_buffer);
            CHECK_CL_SUCCESS(cl_ret);
            cl_ret = unit.ocl_kernel.setArg(idx++, *buffer_);
            CHECK_CL_SUCCESS(cl_ret);
            // src_offset
            cl_ret = unit.ocl_kernel.setArg(idx++, 0);
            CHECK_CL_SUCCESS(cl_ret);
            // dst_offset
            cl_ret = unit.ocl_kernel.setArg(idx++, 0);
            CHECK_CL_SUCCESS(cl_ret);
        }
        return TNN_OK;
    } else {
        uint32_t idx     = SetExecuteUnit2DSizeInfoDefault(unit, dims);

        if (DEVICE_NAIVE == src.GetDeviceType()) {
            cl::Image *image = static_cast<cl::Image *>(dst.GetData());
            cl_ret = unit.ocl_kernel.setArg(idx++, *image);
            CHECK_CL_SUCCESS(cl_ret);
            cl_ret = unit.ocl_kernel.setArg(idx++, *buffer_);
            CHECK_CL_SUCCESS(cl_ret);
            //height
            cl_ret = unit.ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(dims, 2));
            CHECK_CL_SUCCESS(cl_ret);
            //width
            cl_ret = unit.ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(dims, 3));
            CHECK_CL_SUCCESS(cl_ret);
        } else if (DEVICE_OPENCL == src.GetDeviceType()) {
            cl::Image *mat_image = static_cast<cl::Image *>(src.GetData());
            cl_ret               = unit.ocl_kernel.setArg(idx++, *mat_image);
            CHECK_CL_SUCCESS(cl_ret);
            cl_ret = unit.ocl_kernel.setArg(idx++, *buffer_);
            CHECK_CL_SUCCESS(cl_ret);
            //height
            cl_ret = unit.ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(dims, 2));
            CHECK_CL_SUCCESS(cl_ret);
            //width
            cl_ret = unit.ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(dims, 3));
            CHECK_CL_SUCCESS(cl_ret);
        } else {
            return Status(TNNERR_PARAM_ERR, "convert type not support yet");
        }
    }
    return TNN_OK;
}

Status OpenCLMatConverterAcc::SetWarpAffineArgs(OpenCLExecuteUnit& unit, Mat& src, Mat& dst, WarpAffineParam param) {
    MatType mat_type        = src.GetMatType();
    auto output_dims        = dst.GetDims();
    auto input_dims         = src.GetDims();

    uint32_t idx     = SetExecuteUnit2DSizeInfoDefault(unit, output_dims);

    cl_int cl_ret;
    const std::string key = (param.interp_type == INTERP_TYPE_LINEAR) ?
            "WarpAffineLinear" : "WarpAffineNearest";
    if (DEVICE_OPENCL == src.GetDeviceType()) {
        cl::Image *mat_image        = static_cast<cl::Image *>(src.GetData());
        cl::Image *output_mat_image = static_cast<cl::Image *>(dst.GetData());

        // input mat
        cl_ret = execute_map_[key].ocl_kernel.setArg(idx++, *mat_image);
        CHECK_CL_SUCCESS(cl_ret);
        // output mat
        cl_ret = execute_map_[key].ocl_kernel.setArg(idx++, *output_mat_image);
        CHECK_CL_SUCCESS(cl_ret);
        // output height
        cl_ret = execute_map_[key].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(output_dims, 2));
        CHECK_CL_SUCCESS(cl_ret);
        // output width
        cl_ret = execute_map_[key].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(output_dims, 3));
        CHECK_CL_SUCCESS(cl_ret);
        // channel
        cl_ret = execute_map_[key].ocl_kernel.setArg(idx++, UP_DIV(DimsFunctionUtils::GetDim(input_dims, 1), 4));
        CHECK_CL_SUCCESS(cl_ret);
        // input height
        cl_ret = execute_map_[key].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(input_dims, 2));
        CHECK_CL_SUCCESS(cl_ret);
        // input width
        cl_ret = execute_map_[key].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(input_dims, 3));
        CHECK_CL_SUCCESS(cl_ret);
        // inversed transform matrix
        cl_ret = unit.ocl_kernel.setArg(idx++, *matrix_buffer_);
        CHECK_CL_SUCCESS(cl_ret);
        // border_val
        cl_ret = unit.ocl_kernel.setArg(idx++, param.border_val);
        CHECK_CL_SUCCESS(cl_ret);
    } else {
        return Status(TNNERR_PARAM_ERR, "convert type not support yet");
    }
    return TNN_OK;
}

Status OpenCLMatConverterAcc::RunConvertUnit(OpenCLExecuteUnit &unit, cl::CommandQueue *command_queue,
                                              bool need_wait) {
    Status ret = RunKernel(unit.ocl_kernel, unit.global_work_size, unit.local_work_size, command_queue, "MatConvert");
    if (need_wait) {
        //sync
        command_queue->finish();
    }
    return ret;
}

Status OpenCLMatConverterAcc::Resize(Mat& src, Mat& dst, ResizeParam param, void* command_queue) {
    Status ret            = TNN_OK;
    if(src.GetDeviceType() != dst.GetDeviceType()) {
        return Status(TNNERR_PARAM_ERR, "convert type not support yet");
    }
    auto cl_command_queue = static_cast<cl::CommandQueue *>(command_queue);
    if (cl_command_queue == nullptr) {
        LOGE("Get OpenCL command queue failed!\n");
        return Status(TNNERR_NULL_PARAM, "Get OpenCL command queue failed!");
    }
    const std::string key = "Resize";
    OpenCLExecuteUnit unit;
    if(execute_map_.count(key) == 0) {
        std::string program_name = "normalize";
        std::string kernel_name = "";
        if(param.type == INTERP_TYPE_LINEAR) {
            kernel_name = "ResizeBilinear";
        } else if(param.type == INTERP_TYPE_NEAREST) {
            kernel_name = "ResizeNearest";
        } else {
            return Status(TNNERR_PARAM_ERR, "resize type is illegal");
        }
        ret = CreateExecuteUnit(unit, program_name, kernel_name);
        if(ret != TNN_OK) {
            return ret;
        }
        execute_map_[key] = unit; 
    }

    auto dims        = dst.GetDims();
    uint32_t idx     = SetExecuteUnit2DSizeInfoDefault(unit, dims);
    int dst_width    = dst.GetWidth();
    int dst_height   = dst.GetHeight();
    if (dst_width == 0 || dst_height == 0) {
        return Status(TNNERR_INVALID_INPUT, "dst size is zero");
    }
    float w_scale =  ((float)src.GetWidth() / (float)dst_width);
    float h_scale =  ((float)src.GetHeight() / (float)dst_height);
    cl_int cl_ret;
    cl::Image *image_input = static_cast<cl::Image *>(src.GetData());
    cl::Image *image_output = static_cast<cl::Image *>(dst.GetData());
    cl_ret = unit.ocl_kernel.setArg(idx++, *image_input);
    CHECK_CL_SUCCESS(cl_ret);
    cl_ret = unit.ocl_kernel.setArg(idx++, *image_output);
    CHECK_CL_SUCCESS(cl_ret);
    //scale_w
    cl_ret = unit.ocl_kernel.setArg(idx++, w_scale); 
    CHECK_CL_SUCCESS(cl_ret);
    //scale_h
    cl_ret = unit.ocl_kernel.setArg(idx++, h_scale);
    CHECK_CL_SUCCESS(cl_ret);
    //src_w
    cl_ret = unit.ocl_kernel.setArg(idx++, src.GetWidth()); 
    CHECK_CL_SUCCESS(cl_ret);
    //src_h
    cl_ret = unit.ocl_kernel.setArg(idx++, src.GetHeight());
    CHECK_CL_SUCCESS(cl_ret);
    //dst_w
    cl_ret = unit.ocl_kernel.setArg(idx++, dst.GetWidth()); 
    CHECK_CL_SUCCESS(cl_ret);
    //dst_h
    cl_ret = unit.ocl_kernel.setArg(idx++, dst.GetHeight());
    CHECK_CL_SUCCESS(cl_ret);
    ret = RunConvertUnit(unit, cl_command_queue, false);
    if (ret != TNN_OK) {
        return ret;
    }
    return TNN_OK;
}

Status OpenCLMatConverterAcc::Crop(Mat& src, Mat& dst, CropParam param, void* command_queue) {
    Status ret            = TNN_OK;
    if(src.GetDeviceType() != dst.GetDeviceType()) {
        return Status(TNNERR_PARAM_ERR, "convert type not support yet");
    }
    auto cl_command_queue = static_cast<cl::CommandQueue *>(command_queue);
    if (cl_command_queue == nullptr) {
        LOGE("Get OpenCL command queue failed!\n");
        return Status(TNNERR_NULL_PARAM, "Get OpenCL command queue failed!");
    }
    const std::string key = "Crop"; 
    OpenCLExecuteUnit unit;
    if(execute_map_.count(key) == 0) {
        std::string program_name = "copy";
        std::string kernel_name = "Crop";
        ret = CreateExecuteUnit(unit, program_name, kernel_name);
        if(ret != TNN_OK) {
            return ret;
        }
        execute_map_[key] = unit; 
    }

    auto dims        = dst.GetDims();
    uint32_t idx     = SetExecuteUnit2DSizeInfoDefault(unit, dims);

    cl_int cl_ret;

    cl::Image *image_input = static_cast<cl::Image *>(src.GetData());
    cl::Image *image_output = static_cast<cl::Image *>(dst.GetData());
    cl_ret = unit.ocl_kernel.setArg(idx++, *image_input);
    CHECK_CL_SUCCESS(cl_ret);
    cl_ret = unit.ocl_kernel.setArg(idx++, *image_output);
    CHECK_CL_SUCCESS(cl_ret);
    //start_x
    cl_ret = unit.ocl_kernel.setArg(idx++, param.top_left_x); 
    CHECK_CL_SUCCESS(cl_ret);
    //start_y
    cl_ret = unit.ocl_kernel.setArg(idx++, param.top_left_y);
    CHECK_CL_SUCCESS(cl_ret);
    //crop_width
    cl_ret = unit.ocl_kernel.setArg(idx++, param.width); 
    CHECK_CL_SUCCESS(cl_ret);
    //crop_height
    cl_ret = unit.ocl_kernel.setArg(idx++, param.height);
    CHECK_CL_SUCCESS(cl_ret);
    //src_w
    cl_ret = unit.ocl_kernel.setArg(idx++, src.GetWidth()); 
    CHECK_CL_SUCCESS(cl_ret);
    //src_h
    cl_ret = unit.ocl_kernel.setArg(idx++, src.GetHeight());
    CHECK_CL_SUCCESS(cl_ret);

    ret = RunConvertUnit(unit, cl_command_queue, false);
    if (ret != TNN_OK) {
        return ret;
    }
    return TNN_OK;
}

Status OpenCLMatConverterAcc::WarpAffine(Mat& src, Mat& dst, WarpAffineParam param, void* command_queue) {
    Status ret            = TNN_OK;
    auto cl_command_queue = static_cast<cl::CommandQueue *>(command_queue);
    if (cl_command_queue == nullptr) {
        LOGE("Get OpenCL command queue failed!\n");
        return Status(TNNERR_NULL_PARAM, "Get OpenCL command queue failed!");
    }

    if (src.GetData() == nullptr) {
        return Status(TNNERR_NULL_PARAM, "input mat is null");
    }

    if (src.GetDeviceType() != dst.GetDeviceType()) {
        return Status(TNNERR_PARAM_ERR, "src and dst mat type must be same");
    }

    // Init matrix buffer
    auto opencl_runtime   = OpenCLRuntime::GetInstance();
    cl_int ret_cl            = CL_SUCCESS;
    matrix_buffer_size_ = 6 * sizeof(float);
    cl::Buffer* matrix_buffer = new cl::Buffer(*opencl_runtime->Context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                              (cl::size_type)matrix_buffer_size_, nullptr, &ret_cl);
    if (ret_cl != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret_cl)
        if (nullptr != matrix_buffer)
            delete matrix_buffer;
    } else {
        matrix_buffer_.reset(matrix_buffer);
    }

    // Inverse Transform Matrix
    double M[6];
    float m[6];
    M[0] = param.transform[0][0];
    M[1] = param.transform[0][1];
    M[2] = param.transform[0][2];
    M[3] = param.transform[1][0];
    M[4] = param.transform[1][1];
    M[5] = param.transform[1][2];

    double D   = M[0] * M[4] - M[1] * M[3];
    D          = D != 0 ? 1. / D : 0;
    double A11 = M[4] * D, A22 = M[0] * D;
    m[0]      = A11;
    m[1]      = M[1] * (-D);
    m[3]      = M[3] * (-D);
    m[4]      = A22;
    double b1 = -A11 * M[2] - m[1] * M[5];
    double b2 = -m[3] * M[2] - A22 * M[5];
    m[2]      = b1;
    m[5]      = b2;

    // Copy inversed transform matrix to buffer
    cl_int buffer_ret = CL_SUCCESS;
    auto matrix_buffer_ptr =
        cl_command_queue->enqueueMapBuffer(*matrix_buffer_, true, CL_MAP_WRITE, 0, matrix_buffer_size_, nullptr, nullptr, &buffer_ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMMAP_ERROR, "OpenCL MemMap failed");
    }
    memcpy(matrix_buffer_ptr, m, matrix_buffer_size_);
    ret = cl_command_queue->enqueueUnmapMemObject(*matrix_buffer_, matrix_buffer_ptr);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMUNMAP_ERROR, "OpenCL MemUnMap failed");
    }

    // create execute unit
    const std::string key = (param.interp_type == INTERP_TYPE_LINEAR) ?
            "WarpAffineLinear" : "WarpAffineNearest";
    OpenCLExecuteUnit unit;
    if (param.interp_type == INTERP_TYPE_LINEAR && param.border_type == BORDER_TYPE_CONSTANT) {
        if (execute_map_.count(key) == 0) {
            std::string program_name = "warp_affine";
            std::string kernel_name = "WarpAffineLinear";
            ret = CreateExecuteUnit(unit, program_name, kernel_name);
            if (ret != TNN_OK) {
                return ret;
            }
            execute_map_[key] = unit;
        }
    } else if (param.interp_type == INTERP_TYPE_NEAREST && param.border_type == BORDER_TYPE_CONSTANT) {
        if (execute_map_.count(key) == 0) {
            std::string program_name = "warp_affine";
            std::string kernel_name = "WarpAffineNearest";
            ret = CreateExecuteUnit(unit, program_name, kernel_name);
            if (ret != TNN_OK) {
                return ret;
            }
            execute_map_[key] = unit;
        }
    } else {
        return Status(TNNERR_PARAM_ERR, "warpaffine type not support yet");
    }

    ret = SetWarpAffineArgs(unit, src, dst, param);
    if (ret != TNN_OK) {
        return ret;
    }

    ret = RunConvertUnit(unit, cl_command_queue, false);
    if (ret != TNN_OK) {
        return ret;
    }

    return ret;
}

Status OpenCLMatConverterAcc::CvtColor(Mat& src, Mat& dst, ColorConversionType type, void* command_queue) {
    return Status(TNNERR_OPENCL_UNSUPPORT_ERROR, "opencl not support color conversion");
}

Status OpenCLMatConverterAcc::CopyMakeBorder(Mat& src, Mat& dst, CopyMakeBorderParam param, void* command_queue) {
    Status ret            = TNN_OK;
    if(src.GetDeviceType() != dst.GetDeviceType()) {
        return Status(TNNERR_PARAM_ERR, "convert type not support yet");
    }
    auto cl_command_queue = static_cast<cl::CommandQueue *>(command_queue);
    if (cl_command_queue == nullptr) {
        LOGE("Get OpenCL command queue failed!\n");
        return Status(TNNERR_NULL_PARAM, "Get OpenCL command queue failed!");
    }
    const std::string key = "CopyMakeBorder";
    OpenCLExecuteUnit unit;
    if(execute_map_.count(key) == 0) {
        std::string program_name = "copy";
        std::string kernel_name = "CopyMakeBorder";
        ret = CreateExecuteUnit(unit, program_name, kernel_name);
        if(ret != TNN_OK) {
            return ret;
        }
        execute_map_[key] = unit;
    }

    auto dims        = dst.GetDims();
    uint32_t idx     = SetExecuteUnit2DSizeInfoDefault(unit, dims);

    cl_int cl_ret;

    cl::Image *image_input = static_cast<cl::Image *>(src.GetData());
    cl::Image *image_output = static_cast<cl::Image *>(dst.GetData());
    cl_ret = unit.ocl_kernel.setArg(idx++, *image_input);
    CHECK_CL_SUCCESS(cl_ret);
    cl_ret = unit.ocl_kernel.setArg(idx++, *image_output);
    CHECK_CL_SUCCESS(cl_ret);
    // make border top
    cl_ret = unit.ocl_kernel.setArg(idx++, param.top);
    CHECK_CL_SUCCESS(cl_ret);
    // make border left
    cl_ret = unit.ocl_kernel.setArg(idx++, param.left);
    CHECK_CL_SUCCESS(cl_ret);
    // src_w
    cl_ret = unit.ocl_kernel.setArg(idx++, src.GetWidth());
    CHECK_CL_SUCCESS(cl_ret);
    // src_h
    cl_ret = unit.ocl_kernel.setArg(idx++, src.GetHeight());
    CHECK_CL_SUCCESS(cl_ret);
    // src_channel_blocks
    cl_ret = unit.ocl_kernel.setArg(idx++, UP_DIV(src.GetChannel(), 4));
    CHECK_CL_SUCCESS(cl_ret);
    // dst_h
    cl_ret = unit.ocl_kernel.setArg(idx++, dst.GetHeight());
    CHECK_CL_SUCCESS(cl_ret);
    // border_val
    cl_ret = unit.ocl_kernel.setArg(idx++, param.border_val);
    CHECK_CL_SUCCESS(cl_ret);

    ret = RunConvertUnit(unit, cl_command_queue, false);
    if (ret != TNN_OK) {
        return ret;
    }
    return TNN_OK;
}

DECLARE_MAT_CONVERTER_CREATER(OpenCL);
REGISTER_MAT_CONVERTER(OpenCL, DEVICE_OPENCL);
}  // namespace TNN_NS
