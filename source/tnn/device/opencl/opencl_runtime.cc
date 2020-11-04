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

#include "tnn/device/opencl/opencl_runtime.h"
#include "tnn/core/macro.h"

#ifdef SHARING_MEM_WITH_OPENGL
#include <EGL/egl.h>
#endif

namespace TNN_NS {

extern const std::map<std::string, std::vector<unsigned char>> g_opencl_program_map;

static std::mutex g_mtx;

//magic number
static std::map<int, int> AdrenoSubGroup{
    {640, 128}, {630, 128}, {616, 128}, {612, 64}, {610, 64}, {540, 32}, {530, 32},
    {512, 32},  {510, 32},  {509, 32},  {506, 32}, {505, 32}, {405, 32}, {330, 16},
};

std::shared_ptr<OpenCLRuntime> OpenCLRuntime::opencl_runtime_singleton_ = nullptr;
bool OpenCLRuntime::enable_increase_count_ = false;
int OpenCLRuntime::ref_count_              = 0;
bool OpenCLRuntime::init_done_             = false;

OpenCLRuntime *OpenCLRuntime::GetInstance() {
    // don't use DCL
    std::unique_lock<std::mutex> lck(g_mtx);  
    if (nullptr == opencl_runtime_singleton_.get()) {
        opencl_runtime_singleton_.reset(new OpenCLRuntime());
        ref_count_++;
        enable_increase_count_ = false;
    }

    return opencl_runtime_singleton_.get();
}

//if use shared_ptr for opencl runtime and destructor when process finish,
//huawei will crash, so we need increase and decrease ref.

//opencl context will increate ref
void OpenCLRuntime::IncreaseRef() {
    std::unique_lock<std::mutex> lck(g_mtx);
    if (enable_increase_count_) {
        ref_count_++;
    }
    enable_increase_count_ = true;
    LOGD("OpenCLRuntime::IncreaseRef() count=%d\n", ref_count_);
}

//opencl context will decrease ref
void OpenCLRuntime::DecreaseRef() {
    std::unique_lock<std::mutex> lck(g_mtx);
    ref_count_--;
    if (0 == ref_count_) {
        opencl_runtime_singleton_.reset();
        init_done_ = false;
    }
    LOGD("OpenCLRuntime::DecreaseRef() count=%d\n", ref_count_);
}

OpenCLRuntime::OpenCLRuntime() {
    LOGD("OpenCLRuntime() start\n");
    default_build_opts_ = " -cl-mad-enable -cl-fast-relaxed-math -Werror";
}

//Init will get platforms info, get devices info, create opencl context.
Status OpenCLRuntime::Init() {
    std::unique_lock<std::mutex> lck(g_mtx);

    //only init once.
    if (!init_done_) {
        LOGD("Init OpenCL Runtime\n");
        LOGI(
            "OpenCL version: CL_TARGET_OPENCL_VERSION %d   "
            "CL_HPP_TARGET_OPENCL_VERSION %d   CL_HPP_MINIMUM_OPENCL_VERSION "
            "%d\n",
            CL_TARGET_OPENCL_VERSION, CL_HPP_TARGET_OPENCL_VERSION, CL_HPP_MINIMUM_OPENCL_VERSION);

#ifdef TNN_USE_OPENCL_WRAPPER
        if (false == OpenCLSymbols::GetInstance()->LoadOpenCLLibrary()) {
            return Status(TNNERR_DEVICE_LIBRARY_LOAD, "load opencl library falied!");
        }
#endif  // TNN_USE_OPENCL_WRAPPER

        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.size() <= 0) {
            LOGE("OpenCL Platform not found!\n");
            return Status(TNNERR_OPENCL_RUNTIME_ERROR, "OpenCL Platform not found!");
        }

        LOGD("find %lu platforms\n", platforms.size());

        // search GPU
        std::vector<cl::Device> devices;
        for (auto it = platforms.begin(); it != platforms.end(); ++it) {
            std::string platform_name;
            it->getInfo(CL_PLATFORM_NAME, &platform_name);
            it->getDevices(CL_DEVICE_TYPE_GPU, &devices);
            LOGD("platform (%s) has %lu GPUs\n", platform_name.c_str(), devices.size());

            if (devices.size() > 0) {
                std::string device_name = devices[0].getInfo<CL_DEVICE_NAME>();
                LOGD("find GPU: %s\n", device_name.c_str());
                cl::Platform::setDefault(*it);
                break;
            }
        }

        //not found, return error code.
        if (devices.size() <= 0) {
            LOGE("OpenCL Device not found!\n");
            return Status(TNNERR_OPENCL_RUNTIME_ERROR, "OpenCL Device not found!");
        }

        device_                          = std::make_shared<cl::Device>();
        *device_                         = devices[0];
        const std::string device_name    = device_->getInfo<CL_DEVICE_NAME>();
        const std::string device_version = device_->getInfo<CL_DEVICE_VERSION>();
        const std::string opencl_version = device_->getInfo<CL_DEVICE_OPENCL_C_VERSION>();
        LOGD("device name:\t%s\n", device_name.c_str());
        LOGD("opencl version:\t%s\n", device_version.c_str());
        LOGD("highest opencl c version:\t%s\n", opencl_version.c_str());

        gpu_info_ = ParseGpuInfo(device_name, device_version);

        cl_int err;
#if defined(SHARING_MEM_WITH_OPENGL) && (CL_HPP_TARGET_OPENCL_VERSION >= 120)
        // create context from glcontext
        LOGE("Create special opencl context to share with OpenGL\n");
        LOGE("eglGetCurrentContext(): 0x%x\n", eglGetCurrentContext());
        cl_context_properties context_prop[] = {CL_GL_CONTEXT_KHR, (cl_context_properties)eglGetCurrentContext(),
                                                CL_EGL_DISPLAY_KHR, (cl_context_properties)eglGetCurrentDisplay(), 0};
        context_ = std::shared_ptr<cl::Context>(new cl::Context(*device_, context_prop, nullptr, nullptr, &err));

        if (err != CL_SUCCESS) {
            LOGE(
                "Create special opencl context falied, Create common opencl "
                "context then.\n");
            context_ = std::shared_ptr<cl::Context>(new cl::Context(*device_, nullptr, nullptr, nullptr, &err));
        }
#else
        LOGI("Create common opencl context\n");
        context_ = std::shared_ptr<cl::Context>(new cl::Context(*device_, nullptr, nullptr, nullptr, &err));
#endif
        if (err != CL_SUCCESS) {
            LOGE("Context create failed! (ERROR CODE: %d)\n", err);
            return Status(TNNERR_OPENCL_RUNTIME_ERROR, "Context create failed!");
        }

        //get cache size, compute units and frequency.
        device_->getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, &global_memery_cachesize_);
        device_->getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &compute_units_);
        device_->getInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY, &max_freq_);
        device_->getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &local_memory_size_);
        cl_device_fp_config fp_config;
        auto success  = device_->getInfo(CL_DEVICE_HALF_FP_CONFIG, &fp_config);
        support_fp16_ = CL_SUCCESS == success && fp_config > 0;

        LOGD("Global Mem Cache Size: %d\n", (int)global_memery_cachesize_);
        LOGD("Compute Unit: %d\n", (int)compute_units_);
        LOGD("Clock Frequency: %d MHz\n", (int)max_freq_);
        init_done_ = true;
        LOGD("OpenCLRuntime init done!\n");
    }

    return TNN_OK;
}

OpenCLRuntime::~OpenCLRuntime() {
    LOGD("~OpenCLRuntime() start\n");
    program_map_.clear();
    context_.reset();
    device_.reset();
    LOGD("~OpenCLRuntime() end\n");
}

cl::Context *OpenCLRuntime::Context() {
    return context_.get();
}

cl::Device *OpenCLRuntime::Device() {
    return device_.get();
}

uint64_t OpenCLRuntime::DeviceGlobalMemeryCacheSize() const {
    return global_memery_cachesize_;
}

uint32_t OpenCLRuntime::DeviceComputeUnits() const {
    return compute_units_;
}

uint32_t OpenCLRuntime::DeviceMaxFreq() const {
    return max_freq_;
}

uint64_t OpenCLRuntime::DeviceLocalMemerySize() const {
    return local_memory_size_;
}

//get kernel enqueue max work group size 
uint64_t OpenCLRuntime::GetMaxWorkGroupSize(const cl::Kernel &kernel) {
    uint64_t max_workgroup_size = 0;
    int ret                     = kernel.getWorkGroupInfo(*device_, CL_KERNEL_WORK_GROUP_SIZE, &max_workgroup_size);
    if (ret != 0)
        max_workgroup_size = 0;
    return max_workgroup_size;
}

//opencl 2.0 can get SubGroupSize.
uint32_t OpenCLRuntime::GetSubGroupSize(const cl::Kernel &kernel, const cl::NDRange &range) {
    uint32_t sub_group_size = 0;

    if (ADRENO == gpu_info_.type) {
#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_TARGET_OPENCL_VERSION >= 210 && defined(CL_HPP_USE_CL_SUB_GROUPS_KHR)
        cl_int cl_ret;
        sub_group_size = kernel.getSubGroupInfo<CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE>(*device_, range, &cl_ret);
        if (cl_ret != CL_SUCCESS) {
            CHECK_CL_SUCCESS(cl_ret)
            sub_group_size = 0;
        }
#else
        if (AdrenoSubGroup.find(gpu_info_.model_num) != AdrenoSubGroup.end()) {
            sub_group_size = AdrenoSubGroup[gpu_info_.model_num];
        }
#endif
    }

    return sub_group_size;
}

GpuInfo OpenCLRuntime::GetGpuInfo() {
    return gpu_info_;
}

bool OpenCLRuntime::SetPrecision(Precision precision) {
    precision_ = !support_fp16_ ? PRECISION_HIGH : precision;
    return precision_ == precision;
}

Precision OpenCLRuntime::GetPrecision() {
    return precision_;
}

Status OpenCLRuntime::BuildKernel(cl::Kernel &kernel, const std::string &program_name, const std::string &kernel_name,
                                  const std::set<std::string> &build_options) {
    std::string build_options_str;
    bool force_fp32 = false;
    auto it         = build_options.find("-DFORCE_FP32");
    if (it != build_options.end()) {
        force_fp32 = true;
    }
    //set default macro
    if (precision_ != PRECISION_HIGH && !force_fp32) {
        //fp16 enable, kernel will use half and read_imageh and write_imageh.
        LOGD("OpenCL Caucluate Pricision is Half!\n");
        build_options_str =
            "-DFLOAT=half -DFLOAT4=half4 -DRI_F=read_imageh "
            "-DWI_F=write_imageh";
    } else {
        //fp16 not enable, kernel will use float and read_imagef and write_imagef.
        LOGD("OpenCL Caucluate Pricision is Float!\n");
        build_options_str =
            "-DFLOAT=float -DFLOAT4=float4 -DRI_F=read_imagef "
            "-DWI_F=write_imagef";
    }
    for (auto &option : build_options) {
        build_options_str += " " + option;
    }
    build_options_str += default_build_opts_;
    //program identifier = program_name + build_options
    std::string build_program_key = program_name + build_options_str;

    auto build_program_it = program_map_.find(build_program_key);
    cl::Program program;
    //if search program identifier exist, then use it.
    if (build_program_it != program_map_.end()) {
        LOGD("find program: %s\n", build_program_key.c_str());
        program = build_program_it->second;
    } else {
        //load program and build program
        LOGD("build program: %s\n", build_program_key.c_str());
        auto status = this->LoadProgram(program_name, &program);
        if (!status) {
            LOGE("load program (%s) failed!\n", program_name.c_str());
            return Status(TNNERR_OPENCL_KERNELBUILD_ERROR, "load program falied");
        }
        status = this->BuildProgram(build_options_str, &program);
        if (!status) {
            LOGE("%s build failed!\n", program_name.c_str());
            return Status(TNNERR_OPENCL_KERNELBUILD_ERROR, "build program falied");
        }
        program_map_.emplace(build_program_key, program);
    }

    LOGD("build kernel: %s\n", kernel_name.c_str());
    cl_int err;
    kernel = cl::Kernel(program, kernel_name.c_str(), &err);
    if (err != CL_SUCCESS) {
        LOGE("Kernel create failed! (ERROR CODE: %d)\n", err);
        return Status(TNNERR_OPENCL_KERNELBUILD_ERROR, "create kernel falied");
    }
    return TNN_OK;
}

//get gpu divce type
GpuInfo OpenCLRuntime::ParseGpuInfo(std::string device_name, std::string device_version) {
    GpuInfo info;

    if (device_name == "QUALCOMM Adreno(TM)") {
        LOGD("GPU type is ADRENO\n");
        info.type = ADRENO;
        sscanf(device_version.c_str(), "%*s%f%*s%d", &info.opencl_version, &info.model_num);

    } else if (device_name.find("Mali") != std::string::npos) {
        LOGD("GPU type is MALI\n");
        info.type = MALI;
        
        //Mali type MALI-G or MALI_T
        if (device_name.find("Mali-G") != std::string::npos) {
            LOGD("GPU type is MALI-G\n");
            info.type = MALI_G;
            sscanf(device_name.c_str(), "Mali-G%d", &info.model_num);
        } else if (device_name.find("Mali-T") != std::string::npos) {
            LOGD("GPU type is MALI-T\n");
            info.type = MALI_T;
            sscanf(device_name.c_str(), "Mali-T%d", &info.model_num);
        }
        sscanf(device_version.c_str(), "%*s%f%*s", &info.opencl_version);
    }
    LOGD("GPU Type: %d, model_num: %d, opencl version: %f\n", info.type, info.model_num, info.opencl_version);

    return info;
}

//load program with program name.
bool OpenCLRuntime::LoadProgram(const std::string &program_name, cl::Program *program) {
    auto it_source = g_opencl_program_map.find(program_name);
    if (it_source != g_opencl_program_map.end()) {
        cl::Program::Sources sources;
        std::string source(it_source->second.begin(), it_source->second.end());
        sources.push_back(source);
        *program = cl::Program(*Context(), sources);
        return true;
    } else {
        LOGE("Can't find kernel source !\n");
        return false;
    }
}

//build program with build options
bool OpenCLRuntime::BuildProgram(const std::string &build_options, cl::Program *program) {
    cl_int ret = program->build({*device_}, build_options.c_str());
    if (ret != CL_SUCCESS) {
        if (program->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(*device_) == CL_BUILD_ERROR) {
            std::string build_log = program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(*device_);
            LOGE("Program build log: %s \n", build_log.c_str());
        }
        LOGE("Build program failed ! \n");
        return false;
    }
    return true;
}

}  // namespace TNN_NS
