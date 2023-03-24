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
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/string_utils_inner.h"

namespace TNN_NS {


//default constructor will create convert buffer
OpenCLBlobConverterAcc::OpenCLBlobConverterAcc(Blob *blob) : BlobConverterAcc(blob) {
    BlobMemorySizeInfo size_info;
    if (blob->GetBlobDesc().data_format != DATA_FORMAT_NCHW) {
        size_info = Calculate2DCLImageMemorySize(blob->GetBlobDesc());
    } else {
        size_info = Calculate1DMemorySize(blob->GetBlobDesc());
    }
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

    int channel = DimsFunctionUtils::GetDim(blob->GetBlobDesc().dims, 1);
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
            ToString(blob_->GetBlobDesc().data_format) + "_" + ToString(param.reverse_channel) + "_" +
            ToString(do_scale_bias_);
    //create convert unit only once for every key
    if (convert_to_mat_map_.count(to_mat_key) == 0) {
        OpenCLExecuteUnit unit;
        ret = CreateConvertUnit(unit, mat, param, true);
        if (ret != TNN_OK) {
            return ret;
        }
        convert_to_mat_map_[to_mat_key] = unit;
        //only try save once, ignore fail
        auto opencl_runtime   = OpenCLRuntime::GetInstance();
        opencl_runtime->SaveProgramCache();
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
            ToString(blob_->GetBlobDesc().data_format) + "_" + ToString(param.reverse_channel) + "_" +
            ToString(do_scale_bias_);
    //create convert unit only once for every key
    if (convert_from_mat_map_.count(from_mat_key) == 0) {
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

Status OpenCLBlobConverterAcc::GetConvertToMatKernelName(Mat &mat, std::string& kernel_name, std::string& program_name) {
    int dims_size = blob_->GetBlobDesc().dims.size();
    if (blob_->GetBlobDesc().data_type == DATA_TYPE_INT32) {
        if (blob_->GetBlobDesc().data_format == DATA_FORMAT_NHC4W4 && dims_size <= 4) {
            if (NC_INT32 == mat.GetMatType()) {
                kernel_name = "IntBlobConvertToNCINT32";
            } else if (NCHW_FLOAT == mat.GetMatType()) {
                kernel_name = "IntBlobConvertToNCHW";
            } else {
                return Status(TNNERR_PARAM_ERR, "convert type not support yet");
            }
            return TNN_OK;
        } else {
            return Status(TNNERR_PARAM_ERR, "convert type not support yet");
        }
    }
    if (blob_->GetBlobDesc().data_format == DATA_FORMAT_NHC4W4) {
        if (dims_size <= 4) {
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
        } else if (dims_size == 5) {
            if (NCHW_FLOAT == mat.GetMatType()) {
                program_name = "blob_5d_convert_to_mat";
                kernel_name = "Blob5DConvertToNCHW";
            } else {
                char error_str[128];
                sprintf(error_str, "Blob-5D convert type not support mat type: %d",
                        mat.GetMatType());
                return Status(TNNERR_PARAM_ERR, error_str);
            }
        } else if (dims_size == 6) {
            if (NCHW_FLOAT == mat.GetMatType()) {
                program_name = "blob_6d_convert_to_mat";
                kernel_name = "Blob6DConvertToNCHW";
            } else {
                char error_str[128];
                sprintf(error_str, "Blob-6D convert type not support mat type: %d",
                        mat.GetMatType());
                return Status(TNNERR_PARAM_ERR, error_str);
            }
        } else {
            return Status(TNNERR_PARAM_ERR, "convert not support dims > 6");
        }
    }

    if (blob_->GetBlobDesc().data_format == DATA_FORMAT_CNH4) {
        if (NCHW_FLOAT == mat.GetMatType()) {
            kernel_name = "CNH4BlobConvertToNCHW";
        } else {
            return Status(TNNERR_PARAM_ERR, "CNH4 blob convert to mat not support yet");
        }
    }

    if (blob_->GetBlobDesc().data_format == DATA_FORMAT_NCHW) {
        if (NCHW_FLOAT == mat.GetMatType()) {
            kernel_name = "NCHWBlobConvertToNCHW";
        } else {
            return Status(TNNERR_PARAM_ERR, "NCHW blob convert to mat not support yet");
        }
    }

    return TNN_OK;
}

Status OpenCLBlobConverterAcc::GetConvertFromMatKernelName(Mat &mat, std::string& kernel_name, std::string& program_name) {
    int dims_size = blob_->GetBlobDesc().dims.size();
    if (blob_->GetBlobDesc().data_type == DATA_TYPE_INT32) {
        if (blob_->GetBlobDesc().data_format == DATA_FORMAT_NHC4W4 && dims_size <= 4) {
            if (NC_INT32 == mat.GetMatType()) {
                kernel_name = "IntBlobConvertFromNCINT32";
            } else {
                return Status(TNNERR_PARAM_ERR, "convert type not support yet");
            }
            return TNN_OK;
        } else {
            return Status(TNNERR_PARAM_ERR, "convert type not support yet");
        }
    }
    if (blob_->GetBlobDesc().data_format == DATA_FORMAT_NHC4W4) {
        if (dims_size <= 4) {
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
        } else if (dims_size == 5) {
            if (NCHW_FLOAT == mat.GetMatType()) {
                program_name = "blob_5d_convert_from_mat";
                kernel_name = "Blob5DConvertFromNCHW";
            } else {
                char error_str[128];
                sprintf(error_str, "Blob-5D convert type not support mat type: %d",
                        mat.GetMatType());
                return Status(TNNERR_PARAM_ERR, error_str);
            }
        } else if (dims_size == 6) {
            if (NCHW_FLOAT == mat.GetMatType()) {
                program_name = "blob_6d_convert_from_mat";
                kernel_name = "Blob6DConvertFromNCHW";
            } else {
                char error_str[128];
                sprintf(error_str, "Blob-6D convert type not support mat type: %d",
                        mat.GetMatType());
                return Status(TNNERR_PARAM_ERR, error_str);
            }
        } else {
            return Status(TNNERR_PARAM_ERR, "convert not support dims > 6");
        }
    } else {
        if (blob_->GetBlobDesc().data_format == DATA_FORMAT_CNH4) {
            if (NCHW_FLOAT == mat.GetMatType()) {
                kernel_name = "CNH4BlobConvertFromNCHW";
            } else {
                return Status(TNNERR_PARAM_ERR, "CNH4 blob convert from mat not support yet");
            }
        } else {
            char error_str[128];
            sprintf(error_str, "blob convert from mat not support format: %d", blob_->GetBlobDesc().data_format);
            return Status(TNNERR_PARAM_ERR, error_str);
        }
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
        if (DEVICE_NAIVE == mat.GetDeviceType() || DEVICE_ARM == mat.GetDeviceType() ||
            DEVICE_X86 == mat.GetDeviceType()) {
            Status ret = GetConvertToMatKernelName(mat, kernel_name, program_name);
            if (ret != TNN_OK) {
                return ret;
            }
        } else if (DEVICE_OPENCL == mat.GetDeviceType()) {
            if (N8UC4 == mat.GetMatType()) {
                kernel_name = "ConvertToN32FC4Image";
            } else if (NGRAY == mat.GetMatType()) {
                kernel_name = "ConvertToNGray";
            } else {
                return Status(TNNERR_PARAM_ERR, "convert type not support yet");
            }
        } else {
            return Status(TNNERR_PARAM_ERR, "convert type not support yet");
        }
    } else {
        program_name = "convert_from_mat";
        if (DEVICE_NAIVE == mat.GetDeviceType() || DEVICE_ARM == mat.GetDeviceType() ||
            DEVICE_X86 == mat.GetDeviceType()) {
            Status ret = GetConvertFromMatKernelName(mat, kernel_name, program_name);
            if (ret != TNN_OK) {
                return ret;
            }
        } else if (DEVICE_OPENCL == mat.GetDeviceType()) {
            if (N8UC4 == mat.GetMatType()) {
                kernel_name = "ConvertFromN32FC4Image";
            } else if (NGRAY == mat.GetMatType()) {
                kernel_name = "ConvertFromNGray";
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
        if (blob_->GetBlobDesc().data_format == DATA_FORMAT_CNH4) {
            return Status(TNNERR_PARAM_ERR, "cnh4 not support scale and bias yet");
        }
        build_options.emplace("-DENABLE_SCALE_BIAS");
    }

    return CreateExecuteUnit(unit, program_name, kernel_name, build_options);
}

Status OpenCLBlobConverterAcc::SetConvertArgs(OpenCLExecuteUnit &unit, Mat &mat, MatConvertParam param,
                                              bool convert_to_mat) {
    MatType mat_type = mat.GetMatType();
    auto dims        = blob_->GetBlobDesc().dims;

    uint32_t idx     = 0;
    if (blob_->GetBlobDesc().data_format == DATA_FORMAT_NHC4W4) {
        idx = SetExecuteUnit2DSizeInfoDefault(unit, dims);
    } else if (blob_->GetBlobDesc().data_format == DATA_FORMAT_CNH4 &&
               (DEVICE_NAIVE == mat.GetDeviceType() || DEVICE_ARM == mat.GetDeviceType())) {
        idx = SetExecuteUnit2DSizeInfoCNH4(unit, dims);
    } else if (blob_->GetBlobDesc().data_format == DATA_FORMAT_NCHW &&
               (DEVICE_NAIVE == mat.GetDeviceType() || DEVICE_ARM == mat.GetDeviceType() ||
                DEVICE_X86 == mat.GetDeviceType())) {
        idx = SetExecuteUnit2DSizeInfoNCHW(unit, dims);
    } else {
        return Status(TNNERR_PARAM_ERR, "blob data format not support yet");
    }
    cl::Image *image;
    cl::Buffer *buffer;
    if (blob_->GetBlobDesc().data_format != DATA_FORMAT_NCHW) {
        image = static_cast<cl::Image *>(blob_->GetHandle().base);
    } else {
        buffer = static_cast<cl::Buffer *>(blob_->GetHandle().base);
    }

    cl_int cl_ret;
    if (DEVICE_NAIVE == mat.GetDeviceType() || DEVICE_ARM == mat.GetDeviceType() || DEVICE_X86 == mat.GetDeviceType()) {
        if (blob_->GetBlobDesc().data_format != DATA_FORMAT_NCHW) {
            cl_ret = unit.ocl_kernel.setArg(idx++, *image);
            CHECK_CL_SUCCESS(cl_ret);
        } else {
            cl_ret = unit.ocl_kernel.setArg(idx++, *buffer);
            CHECK_CL_SUCCESS(cl_ret);
        }
        cl_ret = unit.ocl_kernel.setArg(idx++, *buffer_);
        CHECK_CL_SUCCESS(cl_ret);
        auto blob_dims_size = blob_->GetBlobDesc().dims.size();
        if (blob_->GetBlobDesc().data_format != DATA_FORMAT_NCHW) {
            cl_ret = unit.ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(dims, 2));
            CHECK_CL_SUCCESS(cl_ret);
            if (blob_->GetBlobDesc().data_format == DATA_FORMAT_NHC4W4) {
                cl_ret = unit.ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(dims, 3));
                CHECK_CL_SUCCESS(cl_ret);
                // set dim4, optional
                if (blob_dims_size >= 5) {
                    cl_ret = unit.ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(dims, 4));
                    CHECK_CL_SUCCESS(cl_ret);
                }
                // set dim5, optional
                if (blob_dims_size >= 6) {
                    cl_ret = unit.ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(dims, 5));
                    CHECK_CL_SUCCESS(cl_ret);
                }
            } else if (blob_->GetBlobDesc().data_format == DATA_FORMAT_CNH4) {
                // batch
                cl_ret = unit.ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(dims, 0));
                CHECK_CL_SUCCESS(cl_ret);
            }
        }

        if (NCHW_FLOAT == mat.GetMatType() || NC_INT32 == mat.GetMatType()) {
            //special for NCHW_FLOAT, need channel parameter
            cl_ret = unit.ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(dims, 1));
            CHECK_CL_SUCCESS(cl_ret);
            cl_ret = unit.ocl_kernel.setArg(idx++, *scale_buffer_);
            CHECK_CL_SUCCESS(cl_ret);
            cl_ret = unit.ocl_kernel.setArg(idx++, *bias_buffer_);
            CHECK_CL_SUCCESS(cl_ret);
        } else {
            // N8UC4 need channel parameter
            if (N8UC4 == mat.GetMatType() && !convert_to_mat) {
                cl_ret = unit.ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(dims, 1));
                CHECK_CL_SUCCESS(cl_ret);
            }
            if (param.scale.size() > 4 || param.bias.size() > 4) {
                return Status(TNNERR_PARAM_ERR, "Cpu convert scale/bias is not valid");
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
        if (NGRAY == mat.GetMatType()) {
            if (blob_->GetBlobDesc().data_format != DATA_FORMAT_NCHW) {
                cl_ret = unit.ocl_kernel.setArg(idx++, *image);
                CHECK_CL_SUCCESS(cl_ret);
            } else {
                cl_ret = unit.ocl_kernel.setArg(idx++, *buffer);
                CHECK_CL_SUCCESS(cl_ret);
            }
            cl::Buffer *mat_buffer  = static_cast<cl::Buffer *>(mat.GetData());
            cl_ret                  = unit.ocl_kernel.setArg(idx++, *mat_buffer);
            CHECK_CL_SUCCESS(cl_ret);
            cl_ret = unit.ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(dims, 2));
            CHECK_CL_SUCCESS(cl_ret);
            cl_ret = unit.ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(dims, 3));
            CHECK_CL_SUCCESS(cl_ret);
        } else {
            cl::Image *mat_image = static_cast<cl::Image *>(mat.GetData());
            cl_ret               = unit.ocl_kernel.setArg(idx++, *mat_image);
            CHECK_CL_SUCCESS(cl_ret);
            cl_ret = unit.ocl_kernel.setArg(idx++, *image);
            CHECK_CL_SUCCESS(cl_ret);
            if (!convert_to_mat) {
                cl_ret = unit.ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(dims, 1));
                CHECK_CL_SUCCESS(cl_ret);
            }
        }
        if (param.scale.size() > 4 || param.bias.size() > 4) {
            return Status(TNNERR_PARAM_ERR, "Gpu convert scale/bias is not valid");
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
    if (nullptr == scale_buffer_ || nullptr == bias_buffer_) {
        LOGE("scale buffer or bias buffer is null\n");
        return Status(TNNERR_OUTOFMEMORY, "scale buffer or bias buffer is null");
    }
    if (nullptr == param.scale.data() || nullptr == param.bias.data()) {
        LOGE("scale or bias is invalid\n");
        return Status(TNNERR_NULL_PARAM, "scale or bias is invalid");
    }

    if (param.scale != host_scale_buffer_) {
        // Copy scale to buffer
        cl_ret = cl_command_queue->enqueueWriteBuffer(*scale_buffer_, CL_TRUE, 0, scale_bias_buffer_size_, param.scale.data());
        CHECK_CL_SUCCESS(cl_ret);
        host_scale_buffer_.assign(param.scale.begin(), param.scale.end());
    }

    if (param.bias != host_bias_buffer_) {
        // Copy bias to buffer
        cl_ret = cl_command_queue->enqueueWriteBuffer(*bias_buffer_, CL_TRUE, 0, scale_bias_buffer_size_, param.bias.data());
        CHECK_CL_SUCCESS(cl_ret);
        host_bias_buffer_.assign(param.bias.begin(), param.bias.end());
    }

    return TNN_OK;
}

DECLARE_BLOB_CONVERTER_CREATER(OpenCL);
REGISTER_BLOB_CONVERTER(OpenCL, DEVICE_OPENCL);

}  // namespace TNN_NS
