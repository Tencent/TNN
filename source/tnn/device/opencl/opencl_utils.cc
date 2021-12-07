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

#include "tnn/device/opencl/opencl_utils.h"

#include <string>
#include <vector>
#ifdef WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "tnn/core/macro.h"
#include "tnn/core/profile.h"
#include "tnn/utils/half_utils_inner.h"
#include "tnn/utils/string_utils.h"

#if (defined __ANDROID_API__) && (__ANDROID_API__ >= 21)
#include <sys/system_properties.h>
#endif

namespace TNN_NS {

// get image width and height
std::vector<int> GetImageShape(const OpenCLMemory *image) {
    std::vector<int> shape;
    cl::Image *cl_image = (cl::Image *)image->GetData();
    size_t width;
    cl_image->getImageInfo(CL_IMAGE_WIDTH, &width);
    size_t height;
    cl_image->getImageInfo(CL_IMAGE_HEIGHT, &height);
    shape.push_back(width);
    shape.push_back(height);
    return shape;
}

// get kernel run time info.
void GetKernelTime(const cl::Event *event, double &kernel_time) {
    cl_int error = CL_SUCCESS;
    error        = event->wait();
    CHECK_CL_SUCCESS(error);
    unsigned long long start_t  = event->getProfilingInfo<CL_PROFILING_COMMAND_START>(&error);
    CHECK_CL_SUCCESS(error);
    unsigned long long end_t    = event->getProfilingInfo<CL_PROFILING_COMMAND_END>(&error);
    CHECK_CL_SUCCESS(error);
    kernel_time  = (end_t - start_t) / 1000000.0;
}

// get kernel run time info.
void GetProfilingTime(const cl::Event *event, double &kernel_time, double &event_queued, double &event_submit,
                      double &event_start, double &event_end) {
    cl_int error = CL_SUCCESS;
    error        = event->wait();
    CHECK_CL_SUCCESS(error);
    unsigned long long queued_t = event->getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>(&error);
    CHECK_CL_SUCCESS(error);
    unsigned long long submit_t = event->getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>(&error);
    CHECK_CL_SUCCESS(error);
    unsigned long long start_t  = event->getProfilingInfo<CL_PROFILING_COMMAND_START>(&error);
    CHECK_CL_SUCCESS(error);
    unsigned long long end_t    = event->getProfilingInfo<CL_PROFILING_COMMAND_END>(&error);
    CHECK_CL_SUCCESS(error);
    kernel_time  = (end_t - start_t) / 1000000.0;
    event_queued = (double)queued_t;
    event_submit = (double)submit_t;
    event_start  = (double)start_t;
    event_end    = (double)end_t;
}

// Run Kernel with 1D, 2D, 3D group size, and local size can be empty.
Status RunKernel(const cl::Kernel &kernel, const std::vector<uint32_t> &gws, const std::vector<uint32_t> &lws,
                 cl::CommandQueue *command_queue, std::string name, OpenCLProfilingData *pdata) {
    LOGD("start RunKernel !\n");

    ASSERT(lws.size() == 0 || lws.size() == gws.size());

    std::vector<uint32_t> internal_global_ws = gws;
    for (size_t i = 0; i < lws.size(); ++i) {
        internal_global_ws[i] = ROUND_UP(gws[i], lws[i]);
    }

    cl::Event event;
    cl_int error = CL_SUCCESS;

    LOGD("gws size: %d, lws size: %d \n", (int)gws.size(), (int)lws.size());

    if (gws.size() == 1) {
        // 1d group size
        if (lws.size() == 0) {
            // local size empty
            error = command_queue->enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(internal_global_ws[0]),
                                                        cl::NullRange, nullptr, &event);
            LOGD("run %s, gws:[%u] , lws: NullRange \n", name.c_str(), internal_global_ws[0]);
        } else {
            error = command_queue->enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(internal_global_ws[0]),
                                                        cl::NDRange(lws[0]), nullptr, &event);
            LOGD("run %s, gws:[%u], lws:[%u] \n", name.c_str(), internal_global_ws[0], lws[0]);
        }
    } else if (gws.size() == 2) {
        // 2d group size
        if (lws.size() == 0) {
            // local size empty
            error = command_queue->enqueueNDRangeKernel(kernel, cl::NullRange,
                                                        cl::NDRange(internal_global_ws[0], internal_global_ws[1]),
                                                        cl::NullRange, nullptr, &event);
            LOGD("run %s, gws:[%u,%u] , lws: NullRange \n", name.c_str(), internal_global_ws[0], internal_global_ws[1]);
        } else {
            error = command_queue->enqueueNDRangeKernel(kernel, cl::NullRange,
                                                        cl::NDRange(internal_global_ws[0], internal_global_ws[1]),
                                                        cl::NDRange(lws[0], lws[1]), nullptr, &event);
            LOGD("run %s, gws:[%u,%u], lws:[%u,%u] \n", name.c_str(), internal_global_ws[0], internal_global_ws[1],
                 lws[0], lws[1]);
        }
    } else {
        // 3d group size
        if (lws.size() == 0) {
            // local size empty
            error = command_queue->enqueueNDRangeKernel(
                kernel, cl::NullRange, cl::NDRange(internal_global_ws[0], internal_global_ws[1], internal_global_ws[2]),
                cl::NullRange, nullptr, &event);
            LOGD("run %s, gws:[%u,%u,%u] , lws: NullRange \n", name.c_str(), internal_global_ws[0],
                 internal_global_ws[1], internal_global_ws[2]);
        } else {
            error = command_queue->enqueueNDRangeKernel(
                kernel, cl::NullRange, cl::NDRange(internal_global_ws[0], internal_global_ws[1], internal_global_ws[2]),
                cl::NDRange(lws[0], lws[1], lws[2]), nullptr, &event);
            LOGD("run %s, gws:[%u,%u,%u], lws:[%u,%u,%u] \n", name.c_str(), internal_global_ws[0],
                 internal_global_ws[1], internal_global_ws[2], lws[0], lws[1], lws[2]);
        }
    }

    if (error != CL_SUCCESS) {
        CHECK_CL_SUCCESS(error);
        return Status(TNNERR_OPENCL_API_ERROR, "OpenCL NDRange failed");
    }

    if (pdata != nullptr) {
        pdata->event = event;
    }
    LOGD("end RunKernel !\n");
    return TNN_OK;
}

bool AdrenoLocalSizeValid(const std::vector<uint32_t> &gws, std::vector<uint32_t>& lws,
                          const uint32_t subgroup_size) {
    return 0 == (lws[0] * lws[1]) % subgroup_size && 0 == gws[0] % lws[0] && 0 == gws[1] % lws[1] &&
           ((lws[0] < lws[1]) == (gws[0] < gws[1]));
}

// adreno local size calculate
std::vector<uint32_t> AdrenoLocalSize2D(const std::vector<uint32_t> &gws, const GpuInfo gpu_info,
                                        const uint32_t compute_units, const uint32_t max_workgroup_size,
                                        const uint32_t subgroup_size) {
    std::vector<uint32_t> lws;
    lws.clear();

    int min_workgroup_count = compute_units;
    // for the later verion gpu 1 SP can process more than one workgroup
    if (gpu_info.model_num >= 540)
        min_workgroup_count = 2 * compute_units;

    // judge gws[1] fisrt
    if (gws[1] % min_workgroup_count == 0) {
        lws.resize(2);
        lws[1] = std::min<uint32_t>(gws[1] / min_workgroup_count, max_workgroup_size);

        // if subgroup size is got, then use it
        if (0 != subgroup_size) {
            int min_workgroup_size = subgroup_size * 2;
            int max_val            = std::max<uint32_t>(max_workgroup_size / lws[1], 1);
            int min_val            = std::max<uint32_t>(min_workgroup_size / lws[1], 1);
            lws[0]                 = std::min<uint32_t>(gws[0], max_val);
            for (; lws[0] >= min_val; lws[0]--) {
                if (AdrenoLocalSizeValid(gws, lws, subgroup_size)) {
                    return lws;
                }
            }
        }

        // another way to calculate lws[0]
        lws[0] = max_workgroup_size / lws[1];
        lws[0] = std::max<uint32_t>(std::min<uint32_t>(gws[0], lws[0]), 1);
        if (0 == gws[0] % lws[0] && 0 == gws[1] % lws[1] && ((lws[0] < lws[1]) == (gws[0] < gws[1]))) {
            return lws;
        }
    }

    // judge gws[0] later
    if (gws[0] % min_workgroup_count == 0) {
        lws.resize(2);
        lws[0] = std::min<uint32_t>(gws[0] / min_workgroup_count, max_workgroup_size);

        // if subgroup size is got, then use it
        if (0 != subgroup_size) {
            int min_workgroup_size = subgroup_size * 2;
            int max_val            = std::max<uint32_t>(max_workgroup_size / lws[0], 1);
            int min_val            = std::max<uint32_t>(min_workgroup_size / lws[0], 1);
            lws[1]                 = std::min<uint32_t>(gws[1], max_val);
            for (; lws[1] >= min_val; lws[1]--) {
                if (0 == (lws[0] * lws[1]) % subgroup_size && 0 == gws[0] % lws[0] && 0 == gws[1] % lws[1] &&
                    ((lws[0] < lws[1]) == (gws[0] < gws[1]))) {
                    return lws;
                }
            }
        }

        // another way to calculate lws[1]
        lws[1] = max_workgroup_size / lws[0];
        lws[1] = std::max<uint32_t>(std::min<uint32_t>(gws[1], lws[1]), 1);
        if (0 == gws[0] % lws[0] && 0 == gws[1] % lws[1] && ((lws[0] < lws[1]) == (gws[0] < gws[1]))) {
            return lws;
        }
    }

    lws.clear();

    return lws;
}

Status AdjustBuildOptionForFp32(std::set<std::string>& build_options)
{
    bool force_fp32 = false;
#if (defined __ANDROID_API__) && (__ANDROID_API__ >= 21)
    char sdk[128] = "0";
    __system_property_get("ro.build.version.sdk", sdk);
    int sdk_version = atoi(sdk);
    // Before Android 8.0, the performance of exp fp16 is poor, so use fp32 instead
    force_fp32 = (sdk_version <= 25);
#elif (defined __ANDROID_API__) && (__ANDROID_API__ < 21)
    force_fp32 = true;
#endif

    if (force_fp32) {
        build_options.emplace("-DFORCE_FP32");
    }

    return TNN_OK;
}

// calculate 3d local size
std::vector<uint32_t> LocalWS3DDefault(OpenCLExecuteUnit &unit) {
    return LocalWS3DDefault(unit.global_work_size, unit.workgroupsize_max, unit.sub_group_size);
}

// adreno will calculate local size, others will return empty local size. the priority is lws[1] > lws[2] > lws[0]
std::vector<uint32_t> LocalWS3DDefault(const std::vector<uint32_t> &gws, const uint32_t max_workgroup_size,
                                       const uint32_t subgroup_size) {
    GpuInfo gpu_info = OpenCLRuntime::GetInstance()->GetGpuInfo();
    std::vector<uint32_t> lws;
    lws.clear();

    if (gpu_info.type == GpuType::ADRENO) {
        uint32_t compute_units = OpenCLRuntime::GetInstance()->DeviceComputeUnits();
        lws.resize(3);
        if (max_workgroup_size == 0) {
            lws[0] = lws[1] = lws[2] = 1;
        } else {
            std::vector<uint32_t> lws_2d_temp;
            lws_2d_temp =
                AdrenoLocalSize2D({gws[1], gws[2]}, gpu_info, compute_units, max_workgroup_size, subgroup_size);

            if (lws_2d_temp.size() != 0) {
                lws[1]                  = lws_2d_temp[0];
                lws[2]                  = lws_2d_temp[1];
                const uint32_t lws_size = lws[1] * lws[2];
                lws[0] = std::max<uint32_t>(max_workgroup_size / lws_size, 1);
                while (gws[0] % lws[0] != 0) {
                    lws[0]--;
                }
            } else {
                lws.clear();
            }
        }
    }

    return lws;
}

// calculate 2d local size
std::vector<uint32_t> LocalWS2DDefault(OpenCLExecuteUnit &unit) {
    return LocalWS2DDefault(unit.global_work_size, unit.workgroupsize_max, unit.sub_group_size);
}

// adreno will calculate local size, others will return empty local size.
std::vector<uint32_t> LocalWS2DDefault(const std::vector<uint32_t> &gws, const uint32_t max_workgroup_size,
                                       const uint32_t subgroup_size) {
    GpuInfo gpu_info = OpenCLRuntime::GetInstance()->GetGpuInfo();
    std::vector<uint32_t> lws;
    if (gpu_info.type == GpuType::ADRENO) {
        uint32_t compute_units = OpenCLRuntime::GetInstance()->DeviceComputeUnits();
        lws.resize(2);
        if (max_workgroup_size == 0) {
            lws[0] = lws[1] = 1;
        } else {
            lws = AdrenoLocalSize2D(gws, gpu_info, compute_units, max_workgroup_size, subgroup_size);
        }
    }

    return lws;
}

std::vector<uint32_t> LocalTune(OpenCLExecuteUnit &unit, OpenCLContext *context, std::string tune_key) {
    std::map<std::string, std::vector<uint32_t>> &tune_map = context->GetLocalSizeTuneMap();
    if (tune_map.count(tune_key) > 0) {
        std::vector<uint32_t> lws = tune_map[tune_key];
        return lws;
    } else {
        cl::CommandQueue *tune_command_queue = context->TuneCommandQueue();
        std::vector<uint32_t> &gws           = unit.global_work_size;
        uint32_t workgroupsize_max           = unit.workgroupsize_max;
        cl::Kernel &kernel                   = unit.ocl_kernel;

        std::vector<uint32_t> opt_lws = unit.local_work_size;
        std::vector<uint32_t> lws(gws.size(), 1);

        double kernel_min_time;
        OpenCLProfilingData data;
        RunKernel(unit.ocl_kernel, unit.global_work_size, unit.local_work_size, tune_command_queue, "tune", &data);
        GetKernelTime(&data.event, kernel_min_time);

        if (gws.size() == 2) {
            for (lws[0] = 1; lws[0] < gws[0] * 2; lws[0] *= 2) {
                for (lws[1] = 1; lws[1] < gws[1] * 2; lws[1] *= 2) {
                    if (lws[0] * lws[1] <= workgroupsize_max) {
                        double kernel_time;
                        RunKernel(unit.ocl_kernel, unit.global_work_size, lws, tune_command_queue, "tune", &data);
                        GetKernelTime(&data.event, kernel_time);
                        if (kernel_time < kernel_min_time) {
                            kernel_min_time = kernel_time;
                            opt_lws.resize(2);
                            opt_lws[0] = lws[0];
                            opt_lws[1] = lws[1];
                        }
                    }
                }
            }
        } else if (gws.size() == 3) {
            for (lws[0] = 1; lws[0] < gws[0] * 2; lws[0] *= 2) {
                for (lws[1] = 1; lws[1] < gws[1] * 2; lws[1] *= 2) {
                    for (lws[2] = 1; lws[2] < gws[2] * 2; lws[2] *= 2) {
                        if (lws[0] * lws[1] * lws[2] <= workgroupsize_max) {
                            double kernel_time;
                            RunKernel(unit.ocl_kernel, unit.global_work_size, lws, tune_command_queue, "tune", &data);
                            GetKernelTime(&data.event, kernel_time);
                            if (kernel_time < kernel_min_time) {
                                kernel_min_time = kernel_time;
                                opt_lws.resize(3);
                                opt_lws[0] = lws[0];
                                opt_lws[1] = lws[1];
                                opt_lws[2] = lws[2];
                            }
                        }
                    }
                }
            }
        }

        // double check
        double kernel_time;
        RunKernel(unit.ocl_kernel, unit.global_work_size, unit.local_work_size, tune_command_queue, "tune", &data);
        GetKernelTime(&data.event, kernel_time);

#ifdef WIN32
        Sleep(10);
#else
        usleep(10000);
#endif

        if (kernel_time < kernel_min_time) {
            tune_map.insert(make_pair(tune_key, unit.local_work_size));
            return unit.local_work_size;
        } else {
            tune_map.insert(make_pair(tune_key, opt_lws));
            return opt_lws;
        }
    }
}


// copy data from clBuffer to clImage.
Status CopyBufferToImage(OpenCLRuntime *runtime, OpenCLContext *context, const cl::Buffer &buffer,
                         const cl::Image &image, int w, int h, bool need_wait) {
    LOGD("start CopyBufferToImage\n");
    std::set<std::string> build_options;
    cl::Kernel kernel;
    std::string kernel_name = "CopyBufferToImage2d";
    Status ret              = runtime->BuildKernel(kernel, "copy_buffer_to_image2d", kernel_name, build_options);
    if (ret != TNN_OK) {
        LOGE("kernel %s build failed!\n", kernel_name.c_str());
        return Status(TNNERR_OPENCL_KERNELBUILD_ERROR, "kernel (CopyBufferToImage2d) build failed!");
    }
    auto status = kernel.setArg(0, buffer);
    ASSERT(status == CL_SUCCESS);
    status = kernel.setArg(1, image);
    ASSERT(status == CL_SUCCESS);
    status = kernel.setArg(2, w);
    ASSERT(status == CL_SUCCESS);
    status = kernel.setArg(3, h);
    ASSERT(status == CL_SUCCESS);
    cl::Event event;
    cl_int error = context->CommandQueue()->enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(w, h, 1),
                                                                 cl::NDRange(1, 1, 1), nullptr, &event);

    if (error != CL_SUCCESS) {
        CHECK_CL_SUCCESS(error);
        return Status(TNNERR_OPENCL_API_ERROR, "OpenCL NDRange failed");
    }

    if (need_wait) {
        event.wait();
    }
    LOGD("end CopyBufferToImage\n");
    return TNN_OK;
}

// copy from clImage to clImage
Status CopyImageToImage(OpenCLRuntime *runtime, OpenCLContext *context, const cl::Image &src, const cl::Image &dst,
                        int w, int h, bool need_wait, OpenCLProfilingData *pdata) {
    LOGD("start CopyImageToImage\n");
    cl::Event event;
    const size_t src_origin[3] = {0, 0, 0};
    const size_t dst_origin[3] = {0, 0, 0};
    const size_t region[3]     = {(size_t)w, (size_t)h, 1};
    cl_int error = context->CommandQueue()->enqueueCopyImage(src, dst, {0, 0, 0}, {0, 0, 0}, {(size_t)w, (size_t)h, 1},
                                                             nullptr, &event);

    if (error != CL_SUCCESS) {
        CHECK_CL_SUCCESS(error);
        return Status(TNNERR_OPENCL_API_ERROR, "OpenCL NDRange failed");
    }

    if (need_wait) {
        event.wait();
    }

#if TNN_PROFILE
    if (pdata != nullptr) {
        pdata->event = event;
    }
#endif

    LOGD("end CopyImageToImage\n");
    return TNN_OK;
}

Status CopyBufferToMat(Mat &mat, cl::Buffer& buffer, DimsVector& dims, const int buffer_size,
                       const MatType& mat_type, cl::CommandQueue *command_queue) {
    int data_type_size = 1;
    if (mat_type == NCHW_FLOAT) {
        data_type_size = sizeof(float);
    } else if (mat_type == NC_INT32) {
        data_type_size = sizeof(int);
    } else if (mat_type == N8UC4) {
        //special for 8UC4, blob channel <= 4.
        dims[1] = 4;
    }
    int size_in_bytes = DimsVectorUtils::Count(dims) * data_type_size;
    if (size_in_bytes > buffer_size) {
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL buffer is smaller than the need!");
    }
    cl_int ret = CL_SUCCESS;
    ret = command_queue->enqueueReadBuffer(buffer, CL_TRUE, 0, size_in_bytes, mat.GetData());
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMUNMAP_ERROR, "OpenCL enqueueReadBuffer failed");
    }

    return TNN_OK;
}

Status CopyMatToBuffer(Mat &mat, cl::Buffer& buffer, DimsVector& dims, const int buffer_size,
                       const MatType& mat_type, cl::CommandQueue *command_queue) {
    int data_type_size = 1;
    if (mat_type == NCHW_FLOAT) {
        data_type_size = sizeof(float);
    } else if (mat_type == NC_INT32) {
        data_type_size = sizeof(int);
    } else if (mat_type == N8UC4) {
        //special for 8UC4, blob channel <= 4.
        dims[1] = 4;
    }
    int size_in_bytes = DimsVectorUtils::Count(dims) * data_type_size;
    if (size_in_bytes > buffer_size) {
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL buffer is smaller than the need!");
    }
    cl_int ret = CL_SUCCESS;
    ret = command_queue->enqueueWriteBuffer(buffer, CL_TRUE, 0, size_in_bytes, mat.GetData());
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMUNMAP_ERROR, "OpenCL enqueueWriteBuffer failed");
    }

    return TNN_OK;
}

uint32_t gcd(uint32_t number1, uint32_t number2) {
    return number2 == 0 ? number1 : gcd(number2, number1 % number2);
}

// create execute unit with kernel name and build options.
Status CreateExecuteUnit(OpenCLExecuteUnit &unit, const std::string &program_name, const std::string &kernel_name,
                         const std::set<std::string> &build_opt) {
    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();

    unit.program_name = program_name;
    unit.kernel_name = kernel_name;

    if (opencl_runtime == NULL) {
        LOGE("CreateExecuteUnit Get OpenCLRuntime failed!");
        return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "CreateExecuteUnit Get OpenCLRuntime failed!");
    }
    Status ret = opencl_runtime->BuildKernel(unit.ocl_kernel, program_name, kernel_name, build_opt);
    if (ret != TNN_OK) {
        LOGE("kernel (%s) build failed!\n", kernel_name.c_str());
        return ret;
    }
    unit.workgroupsize_max = static_cast<uint32_t>(opencl_runtime->GetMaxWorkGroupSize(unit.ocl_kernel));
    if (unit.workgroupsize_max == 0) {
        LOGE("Get max workgroup size failed!\n");
        return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "Get max workgroup size failed!");
    }

    unit.sub_group_size = static_cast<uint32_t>(opencl_runtime->GetSubGroupSize(unit.ocl_kernel));

    unit.local_mem_size = opencl_runtime->DeviceLocalMemerySize();

    return TNN_OK;
}

// set execute unit 3d default global size, local size and kernel arguments.
uint32_t SetExecuteUnit3DSizeInfoDefault(OpenCLExecuteUnit &unit, DimsVector dims) {
    uint32_t gws0 = 0, gws1 = 0, gws2;
    if (dims.size() == 5) {
        // dim4
        gws0 = DimsFunctionUtils::GetDim(dims, 4);
        // channel-blocks/4
        gws1 = UP_DIV(DimsFunctionUtils::GetDim(dims, 1), 4);
        // batch * dim2 * dim3
        gws2 = DimsFunctionUtils::GetDim(dims, 0) * DimsFunctionUtils::GetDim(dims, 2) *
               DimsFunctionUtils::GetDim(dims, 3);
    } else if (dims.size() == 6) {
        // dim4 * dim5
        gws0 = DimsFunctionUtils::GetDim(dims, 4) * DimsFunctionUtils::GetDim(dims, 5);
        // channel-blocks/4
        gws1 = UP_DIV(DimsFunctionUtils::GetDim(dims, 1), 4);
        // batch * dim2 * dim3
        gws2 = DimsFunctionUtils::GetDim(dims, 0) * DimsFunctionUtils::GetDim(dims, 2) *
               DimsFunctionUtils::GetDim(dims, 3);
    } else {
        // width
        gws0 = DimsFunctionUtils::GetDim(dims, 3);
        // channel-blocks/4
        gws1 = UP_DIV(DimsFunctionUtils::GetDim(dims, 1), 4);
        // batch * height
        gws2 = DimsFunctionUtils::GetDim(dims, 0) * DimsFunctionUtils::GetDim(dims, 2);
    }
    unit.global_work_size = {gws0, gws1, gws2};

    // change the order temporarily to get the local size
    std::vector<uint32_t> temp_gws = {unit.global_work_size[1], unit.global_work_size[0], unit.global_work_size[2]};
    std::vector<uint32_t> temp_lws = LocalWS3DDefault(temp_gws, unit.workgroupsize_max, unit.sub_group_size);

    if (3 == temp_lws.size())
        unit.local_work_size = {temp_lws[1], temp_lws[0], temp_lws[2]};
    else
        unit.local_work_size.clear();

    uint32_t idx = 0;
    unit.ocl_kernel.setArg(idx++, unit.global_work_size[0]);
    unit.ocl_kernel.setArg(idx++, unit.global_work_size[1]);
    unit.ocl_kernel.setArg(idx++, unit.global_work_size[2]);

    return idx;
}

// set execute unit 2d default global size, local size and kernel arguments.
uint32_t SetExecuteUnit2DSizeInfoDefault(OpenCLExecuteUnit &unit, DimsVector dims) {
    uint32_t image_width = 0, image_height = 0;
    if (dims.size() == 5) {
        // channel-blocks * dim4
        image_width = UP_DIV(DimsFunctionUtils::GetDim(dims, 1), 4) * DimsFunctionUtils::GetDim(dims, 4);
        // batch * dim2 * dim3
        image_height = DimsFunctionUtils::GetDim(dims, 0) * DimsFunctionUtils::GetDim(dims, 2) *
                       DimsFunctionUtils::GetDim(dims, 3);
    } else if (dims.size() == 6) {
        // channel-blocks * dim4 * dim5
        image_width = UP_DIV(DimsFunctionUtils::GetDim(dims, 1), 4) * DimsFunctionUtils::GetDim(dims, 4) *
                      DimsFunctionUtils::GetDim(dims, 5);
        // batch * dim2 * dim3
        image_height = DimsFunctionUtils::GetDim(dims, 0) * DimsFunctionUtils::GetDim(dims, 2) *
                       DimsFunctionUtils::GetDim(dims, 3);
    } else {
        // channel-blocks * [width]
        image_width = UP_DIV(DimsFunctionUtils::GetDim(dims, 1), 4) * DimsFunctionUtils::GetDim(dims, 3);
        // batch * height
        image_height = DimsFunctionUtils::GetDim(dims, 0) * DimsFunctionUtils::GetDim(dims, 2);
    }
    unit.global_work_size = {image_width, image_height};
    unit.local_work_size = LocalWS2DDefault(unit);
    uint32_t idx         = 0;
    unit.ocl_kernel.setArg(idx++, unit.global_work_size[0]);
    unit.ocl_kernel.setArg(idx++, unit.global_work_size[1]);
    return idx;
}

// set execute unit 2d global size for cnh4, local size and kernel arguments.
uint32_t SetExecuteUnit2DSizeInfoCNH4(OpenCLExecuteUnit &unit, DimsVector dims) {
    unit.global_work_size = {
        // height-blocks
        static_cast<uint32_t>(UP_DIV(DimsFunctionUtils::GetDim(dims, 2), 4)),
        // channel * batch
        static_cast<uint32_t>(DimsFunctionUtils::GetDim(dims, 1) * DimsFunctionUtils::GetDim(dims, 0)),
    };
    unit.local_work_size = LocalWS2DDefault(unit);
    uint32_t idx         = 0;
    unit.ocl_kernel.setArg(idx++, unit.global_work_size[0]);
    unit.ocl_kernel.setArg(idx++, unit.global_work_size[1]);
    return idx;
}

// set execute unit 2d global size for nchw, local size and kernel arguments.
uint32_t SetExecuteUnit2DSizeInfoNCHW(OpenCLExecuteUnit &unit, DimsVector dims) {
    int count = DimsVectorUtils::Count(dims, 2);
    uint32_t gws0 = count == 0 ? 1 : count;
    unit.global_work_size = {
        // [dim2 * dim3 ...]
        gws0,
        // batch * channel
        static_cast<uint32_t>(DimsFunctionUtils::GetDim(dims, 0) * DimsFunctionUtils::GetDim(dims, 1)),
    };
    unit.local_work_size = LocalWS2DDefault(unit);
    uint32_t idx         = 0;
    unit.ocl_kernel.setArg(idx++, unit.global_work_size[0]);
    unit.ocl_kernel.setArg(idx++, unit.global_work_size[1]);
    return idx;
}

// set execute unit 1d default global size, local size and kernel arguments.
uint32_t SetExecuteUnit1DSizeInfoDefault(OpenCLExecuteUnit &unit, DimsVector dims) {
    unit.global_work_size = {(uint32_t)DimsVectorUtils::Count(dims)};
    unit.local_work_size = {unit.workgroupsize_max};
    uint32_t idx         = 0;
    unit.ocl_kernel.setArg(idx++, unit.global_work_size[0]);
    return idx;
}


}  // namespace TNN_NS
