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

#include "tnn/device/opencl/opencl_blob_converter.h"
#include "tnn/core/macro.h"
#include "tnn/device/opencl/opencl_utils.h"
#include "tnn/memory_manager/blob_memory_size_info.h"
#include "tnn/utils/blob_memory_size_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/string_utils_inner.h"

namespace TNN_NS {


//default contructor will create convert buffer
OpenCLBlobConverterAcc::OpenCLBlobConverterAcc(Blob *blob) : BlobConverterAcc(blob) {
    BlobMemorySizeInfo size_info = Calculate2DCLImageMemorySize(blob->GetBlobDesc());
    // force float to get the max memeory
    size_info.data_type   = DATA_TYPE_FLOAT;  
    auto opencl_runtime   = OpenCLRuntime::GetInstance();
    buffer_size_          = GetBlobMemoryBytesSize(size_info);
    cl_int ret            = CL_SUCCESS;
    cl::Buffer* cl_buffer = new cl::Buffer(*opencl_runtime->Context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                            (cl::size_type)buffer_size_, nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        if (nullptr != cl_buffer)
            delete cl_buffer;
    } else {
        buffer_.reset(cl_buffer);
    }

    int channel = blob->GetBlobDesc().dims[1];
    scale_bias_buffer_size_ = channel * sizeof(float);
    cl::Buffer* scale_buffer = new cl::Buffer(*opencl_runtime->Context(),
                                              CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                              (cl::size_type)scale_bias_buffer_size_, nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        if (nullptr != scale_buffer)
            delete scale_buffer;
    } else {
        scale_buffer_.reset(scale_buffer);
    }

    cl::Buffer* bias_buffer = new cl::Buffer(*opencl_runtime->Context(),
                                             CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                             (cl::size_type)scale_bias_buffer_size_, nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        if (nullptr != bias_buffer)
            delete bias_buffer;
    } else {
        bias_buffer_.reset(bias_buffer);
    }
}

OpenCLBlobConverterAcc::~OpenCLBlobConverterAcc() {
    buffer_.reset();
    scale_buffer_.reset();
    bias_buffer_.reset();
}

//convert blob data to mat async
Status OpenCLBlobConverterAcc::ConvertToMatAsync(Mat &mat, MatConvertParam param, void *command_queue) {
    if (nullptr == buffer_) {
        LOGE("OpenCLBlobConverter buffer allocate failed\n");
        return Status(TNNERR_NULL_PARAM, "OpenCLBlobConverter buffer allocate failed!");
    }

    Status ret            = TNN_OK;
    auto cl_command_queue = static_cast<cl::CommandQueue *>(command_queue);
    if (cl_command_queue == nullptr) {
        LOGE("Get OpenCL command queue failed!\n");
        return Status(TNNERR_NULL_PARAM, "Get OpenCL command queue failed!");
    }

    do_scale_bias_  = NeedDoScaleBias(param);
    //create identifier
    std::string to_mat_key = ToString(mat.GetDeviceType()) + "_" + ToString(mat.GetMatType()) + "_" +
                            ToString(param.reverse_channel) + "_" + ToString(do_scale_bias_);
    //create convert unit only once for every key
    if (convert_to_mat_map_.count(to_mat_key) == 0) {
        OpenCLExecuteUnit unit;
        ret = CreateConvertUnit(unit, mat, param, true);
        if (ret != TNN_OK) {
            return ret;
        }
        convert_to_mat_map_[to_mat_key] = unit;
    }

    OpenCLExecuteUnit unit = convert_to_mat_map_[to_mat_key];
    // set arguments
    ret                    = SetConvertArgs(unit, mat, param, true);
    if (ret != TNN_OK) {
        return ret;
    }

    // if mat type is nchw_float, need copy scale and bias to buffer
    if (mat.GetMatType() == NCHW_FLOAT) {
        ret = CopyScaleBiasToBuffer(param, cl_command_queue);
        if (ret != TNN_OK) {
            return ret;
        }
    }

    // run convert unit
    ret = RunConvertUnit(unit, cl_command_queue, false);
    if (ret != TNN_OK) {
        return ret;
    }

    //if mat device is cpu, need convert blob to buffer and copy buffer data to mat
    if (mat.GetDeviceType() != DEVICE_OPENCL) {
        ret = CopyBufferDataToMat(mat, cl_command_queue);
        if (ret != TNN_OK) {
            return ret;
        }
    }

    return ret;
}

//convert mat data to blob async
Status OpenCLBlobConverterAcc::ConvertFromMatAsync(Mat &mat, MatConvertParam param, void *command_queue) {
    if (nullptr == buffer_) {
        LOGE("OpenCLBlobConverter buffer allocate failed\n");
        return Status(TNNERR_NULL_PARAM, "OpenCLBlobConverter buffer allocate failed!");
    }

    Status ret            = TNN_OK;
    auto cl_command_queue = static_cast<cl::CommandQueue *>(command_queue);
    if (cl_command_queue == nullptr) {
        LOGE("Get OpenCL command queue failed!\n");
        return Status(TNNERR_NULL_PARAM, "Get OpenCL command queue failed!");
    }

    do_scale_bias_  = NeedDoScaleBias(param);
    //create identifier
    std::string from_mat_key = ToString(mat.GetDeviceType()) + "_" + ToString(mat.GetMatType()) + "_" +
                                ToString(param.reverse_channel) + "_" + ToString(do_scale_bias_);
    //create convert unit only once for every key
    if (convert_to_mat_map_.count(from_mat_key) == 0) {
        OpenCLExecuteUnit unit;
        ret = CreateConvertUnit(unit, mat, param, false);
        if (ret != TNN_OK) {
            return ret;
        }
        convert_from_mat_map_[from_mat_key] = unit;
    }

    OpenCLExecuteUnit unit = convert_from_mat_map_[from_mat_key];
    // set arguments
    ret                    = SetConvertArgs(unit, mat, param, false);
    if (ret != TNN_OK) {
        return ret;
    }

    //if mat device is cpu, need copy mat data to buffer and convert buffer to blob 
    if (mat.GetDeviceType() != DEVICE_OPENCL) {
        ret = CopyMatToBufferData(mat, cl_command_queue);
        if (ret != TNN_OK) {
            return ret;
        }
    }

    // if mat type is nchw_float, need copy scale and bias to buffer
    if (mat.GetMatType() == NCHW_FLOAT) {
        ret = CopyScaleBiasToBuffer(param, cl_command_queue);
        if (ret != TNN_OK) {
            return ret;
        }
    }
    // run convert unit
    ret = RunConvertUnit(unit, cl_command_queue, false);
    if (ret != TNN_OK) {
        return ret;
    }

    return ret;
}

Status OpenCLBlobConverterAcc::ConvertToMat(Mat &mat, MatConvertParam param, void *command_queue) {
    Status ret = ConvertToMatAsync(mat, param, command_queue);
    //sync
    if (ret == TNN_OK) {
        cl::CommandQueue *opencl_command_queue = static_cast<cl::CommandQueue *>(command_queue);
        opencl_command_queue->finish();
    }
    return ret;
}

Status OpenCLBlobConverterAcc::ConvertFromMat(Mat &mat, MatConvertParam param, void *command_queue) {
    Status ret = ConvertFromMatAsync(mat, param, command_queue);
    //sync
    if (ret == TNN_OK) {
        cl::CommandQueue *opencl_command_queue = static_cast<cl::CommandQueue *>(command_queue);
        opencl_command_queue->finish();
    }
    return ret;
}

bool OpenCLBlobConverterAcc::NeedDoScaleBias(MatConvertParam &param) {
    for (auto s : param.scale) {
        if (s != 1.0f) {
            return true;
        }
    }
    for (auto b : param.bias) {
        if (b != 0.0f) {
            return true;
        }
    }

    return false;
}

Status OpenCLBlobConverterAcc::GetConvertToMatKernelName(Mat &mat, std::string& kernel_name) {
    if (N8UC3 == mat.GetMatType()) {
        kernel_name = "ConvertToN8UC3";
    } else if (N8UC4 == mat.GetMatType()) {
        kernel_name = "ConvertToN8UC4";
    } else if (NGRAY == mat.GetMatType()) {
        kernel_name = "ConvertToNGray";
    } else if (NCHW_FLOAT == mat.GetMatType()) {
        kernel_name = "ConvertToNCHW";
    } else {
        return Status(TNNERR_PARAM_ERR, "convert type not support yet");
    }

    return TNN_OK;
}

Status OpenCLBlobConverterAcc::GetConvertFromMatKernelName(Mat &mat, std::string& kernel_name) {
    if (N8UC3 == mat.GetMatType()) {
        kernel_name = "ConvertFromN8UC3";
    } else if (N8UC4 == mat.GetMatType()) {
        kernel_name = "ConvertFromN8UC4";
    } else if (NGRAY == mat.GetMatType()) {
        kernel_name = "ConvertFromNGray";
    } else if (NNV21 == mat.GetMatType()) {
        kernel_name = "ConvertFromNNV21";
    } else if (NCHW_FLOAT == mat.GetMatType()) {
        kernel_name = "ConvertFromNCHW";
    } else {
        return Status(TNNERR_PARAM_ERR, "convert type not support yet");
    }

    return TNN_OK;
}

//CreateConvertUnit select kernel name and create execute unit
Status OpenCLBlobConverterAcc::CreateConvertUnit(OpenCLExecuteUnit &unit, Mat &mat, MatConvertParam param,
                                                 bool convert_to_mat) {
    std::set<std::string> build_options;
    std::string program_name = "";
    std::string kernel_name  = "";
    if (convert_to_mat) {
        program_name = "convert_to_mat";
        //DEVICE_NAIVE AND DEVICE_ARM is same for memory type.
        if (DEVICE_NAIVE == mat.GetDeviceType() || DEVICE_ARM == mat.GetDeviceType()) {
            Status ret = GetConvertToMatKernelName(mat, kernel_name);
            if (ret != TNN_OK) {
                return ret;
            }
        } else if (DEVICE_OPENCL == mat.GetDeviceType()) {
            if (N8UC4 == mat.GetMatType()) {
                kernel_name = "ConvertToN32FC4Image";
            } else {
                return Status(TNNERR_PARAM_ERR, "convert type not support yet");
            }
        } else {
            return Status(TNNERR_PARAM_ERR, "convert type not support yet");
        }
    } else {
        program_name = "convert_from_mat";
        if (DEVICE_NAIVE == mat.GetDeviceType() || DEVICE_ARM == mat.GetDeviceType()) {
            Status ret = GetConvertFromMatKernelName(mat, kernel_name);
            if (ret != TNN_OK) {
                return ret;
            }
        } else if (DEVICE_OPENCL == mat.GetDeviceType()) {
            if (N8UC4 == mat.GetMatType()) {
                kernel_name = "ConvertFromN32FC4Image";
            } else {
                return Status(TNNERR_PARAM_ERR, "convert type not support yet");
            }
        } else {
            return Status(TNNERR_PARAM_ERR, "convert type not support yet");
        }
    }

    if (param.reverse_channel) {
        build_options.emplace("-DSWAP_RB");
    }

    if (do_scale_bias_) {
        build_options.emplace("-DENABLE_SCALE_BIAS");
    }

    return CreateExecuteUnit(unit, program_name, kernel_name, build_options);
}

Status OpenCLBlobConverterAcc::SetConvertArgs(OpenCLExecuteUnit &unit, Mat &mat, MatConvertParam param,
                                              bool convert_to_mat) {
    MatType mat_type = mat.GetMatType();
    auto dims        = blob_->GetBlobDesc().dims;

    uint32_t idx     = SetExecuteUnit2DSizeInfoDefault(unit, dims);
    cl::Image *image = static_cast<cl::Image *>(blob_->GetHandle().base);

    cl_int cl_ret;
    if (DEVICE_NAIVE == mat.GetDeviceType() || DEVICE_ARM == mat.GetDeviceType()) {
        cl_ret = unit.ocl_kernel.setArg(idx++, *image);
        CHECK_CL_SUCCESS(cl_ret);
        cl_ret = unit.ocl_kernel.setArg(idx++, *buffer_);
        CHECK_CL_SUCCESS(cl_ret);
        //height
        cl_ret = unit.ocl_kernel.setArg(idx++, dims[2]); 
        CHECK_CL_SUCCESS(cl_ret);
        //width
        cl_ret = unit.ocl_kernel.setArg(idx++, dims[3]);
        CHECK_CL_SUCCESS(cl_ret);
        if (NCHW_FLOAT == mat.GetMatType()) {
            //special for NCHW_FLOAT, need channel parameter
            cl_ret = unit.ocl_kernel.setArg(idx++, dims[1]);
            CHECK_CL_SUCCESS(cl_ret);
            cl_ret = unit.ocl_kernel.setArg(idx++, *scale_buffer_);
            CHECK_CL_SUCCESS(cl_ret);
            cl_ret = unit.ocl_kernel.setArg(idx++, *bias_buffer_);
            CHECK_CL_SUCCESS(cl_ret);
        } else {
            // N8UC4 need channel parameter
            if (N8UC4 == mat.GetMatType() && !convert_to_mat) {
                cl_ret = unit.ocl_kernel.setArg(idx++, dims[1]);
                CHECK_CL_SUCCESS(cl_ret);
            }
            // pad scale && bias for vectors in opencl kernel
            while (param.scale.size() < 4) {
                param.scale.push_back(1.0f);
            }
            while (param.bias.size() < 4) {
                param.bias.push_back(0.0f);
            }
            cl_ret = unit.ocl_kernel.setArg(idx++, sizeof(float) * param.scale.size(), param.scale.data());
            CHECK_CL_SUCCESS(cl_ret);
            cl_ret = unit.ocl_kernel.setArg(idx++, sizeof(float) * param.bias.size(), param.bias.data());
            CHECK_CL_SUCCESS(cl_ret);
        }
    } else if (DEVICE_OPENCL == mat.GetDeviceType()) {
        cl::Image *mat_image = static_cast<cl::Image *>(mat.GetData());
        cl_ret               = unit.ocl_kernel.setArg(idx++, *mat_image);
        CHECK_CL_SUCCESS(cl_ret);
        cl_ret = unit.ocl_kernel.setArg(idx++, *image);
        CHECK_CL_SUCCESS(cl_ret);
        if (!convert_to_mat) {
            cl_ret = unit.ocl_kernel.setArg(idx++, dims[1]);
            CHECK_CL_SUCCESS(cl_ret);
        }
        cl_ret = unit.ocl_kernel.setArg(idx++, sizeof(float) * param.scale.size(), param.scale.data());
        CHECK_CL_SUCCESS(cl_ret);
        cl_ret = unit.ocl_kernel.setArg(idx++, sizeof(float) * param.bias.size(), param.bias.data());
        CHECK_CL_SUCCESS(cl_ret);
    } else {
        return Status(TNNERR_PARAM_ERR, "convert type not support yet");
    }
    return TNN_OK;
}

Status OpenCLBlobConverterAcc::RunConvertUnit(OpenCLExecuteUnit &unit, cl::CommandQueue *command_queue,
                                              bool need_wait) {
    Status ret = RunKernel(unit.ocl_kernel, unit.global_work_size, unit.local_work_size, command_queue, "BlobConvert");
    if (need_wait) {
        //sync
        command_queue->finish();
    }
    return ret;
}

//enqueueMapBuffer get cpu buffer pointer, copy buffer pointer to mat, enqueueUnmapMemObject.
Status OpenCLBlobConverterAcc::CopyBufferDataToMat(Mat &mat, cl::CommandQueue *command_queue) {
    MatType mat_type   = mat.GetMatType();
    DimsVector dims    = blob_->GetBlobDesc().dims;

    Status ret = CopyBufferToMat(mat, *buffer_, dims, buffer_size_, mat_type, command_queue);
    if (ret != TNN_OK) {
        return ret;
    }

    return TNN_OK;
}

//enqueueMapBuffer get cpu buffer pointer, copy mat to buffer pointer, enqueueUnmapMemObject.
Status OpenCLBlobConverterAcc::CopyMatToBufferData(Mat &mat, cl::CommandQueue *command_queue) {
    MatType mat_type   = mat.GetMatType();
    DimsVector dims    = blob_->GetBlobDesc().dims;

    Status ret = CopyMatToBuffer(mat, *buffer_, dims, buffer_size_, mat_type, command_queue);
    if (ret != TNN_OK) {
        return ret;
    }

    return TNN_OK;
}

Status OpenCLBlobConverterAcc::CopyScaleBiasToBuffer(MatConvertParam param, cl::CommandQueue *cl_command_queue) {
    cl_int cl_ret;
    // Copy scale and bias to buffer
    auto scale_buffer_ptr =
        cl_command_queue->enqueueMapBuffer(*scale_buffer_, true, CL_MAP_WRITE, 0, scale_bias_buffer_size_, nullptr, nullptr, &cl_ret);
    CHECK_CL_SUCCESS(cl_ret);
    memcpy(scale_buffer_ptr, param.scale.data(), scale_bias_buffer_size_);
    cl_ret = cl_command_queue->enqueueUnmapMemObject(*scale_buffer_, scale_buffer_ptr);
    CHECK_CL_SUCCESS(cl_ret);

    auto bias_buffer_ptr =
        cl_command_queue->enqueueMapBuffer(*bias_buffer_, true, CL_MAP_WRITE, 0, scale_bias_buffer_size_, nullptr, nullptr, &cl_ret);
    CHECK_CL_SUCCESS(cl_ret);
    memcpy(bias_buffer_ptr, param.bias.data(), scale_bias_buffer_size_);
    cl_ret = cl_command_queue->enqueueUnmapMemObject(*bias_buffer_, bias_buffer_ptr);
    CHECK_CL_SUCCESS(cl_ret);

    return TNN_OK;
}

DECLARE_BLOB_CONVERTER_CREATER(OpenCL);
REGISTER_BLOB_CONVERTER(OpenCL, DEVICE_OPENCL);

}  // namespace TNN_NS
