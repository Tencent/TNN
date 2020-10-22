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

#ifndef TNN_SOURCE_TNN_DEVICE_OPENCL_OPENCL_RUNTIME_H_
#define TNN_SOURCE_TNN_DEVICE_OPENCL_OPENCL_RUNTIME_H_

#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include "tnn/core/status.h"
#include "tnn/core/common.h"
#include "tnn/device/opencl/opencl_wrapper.h"

namespace TNN_NS {

enum GpuType { OTHER = 0, ADRENO = 1, MALI = 2, MALI_T = 3, MALI_G = 4 };

struct GpuInfo {
    GpuType type = OTHER;
    int model_num = 0;
    float opencl_version = 0;
};

// Base GPU cache size used for computing local work group size.
const int32_t g_base_gpu_mem_cachesize = 16384;

class OpenCLRuntime {
public:
    static OpenCLRuntime *GetInstance();
    static void IncreaseRef();
    static void DecreaseRef();

    ~OpenCLRuntime();
    OpenCLRuntime(const OpenCLRuntime &) = delete;
    OpenCLRuntime &operator=(const OpenCLRuntime &) = delete;

    Status Init();

    cl::Context *Context();
    cl::Device *Device();
    uint64_t DeviceGlobalMemeryCacheSize() const;
    uint32_t DeviceComputeUnits() const;
    uint32_t DeviceMaxFreq() const;
    uint64_t DeviceLocalMemerySize() const;
    uint64_t GetMaxWorkGroupSize(const cl::Kernel &kernel);
    uint32_t GetSubGroupSize(const cl::Kernel &kernel, const cl::NDRange &range = cl::NullRange);
    GpuInfo GetGpuInfo();
    bool GetFp16Enable() const;
    bool SetFp16Enable(bool enable);
    void SetPrecision(Precision precision);

    Status BuildKernel(cl::Kernel &kernel, const std::string &program_name, const std::string &kernel_name,
                       const std::set<std::string> &build_options);

private:
    OpenCLRuntime();
    GpuInfo ParseGpuInfo(std::string device_name, std::string device_version);

    bool LoadProgram(const std::string &program_name, cl::Program *program);
    bool BuildProgram(const std::string &build_options, cl::Program *program);

private:
    static std::shared_ptr<OpenCLRuntime> opencl_runtime_singleton_;
    static bool enable_increase_count_;
    static int ref_count_;
    static bool init_done_;

    std::shared_ptr<cl::Context> context_ = nullptr;
    std::shared_ptr<cl::Device> device_ = nullptr;
    std::map<std::string, cl::Program> program_map_ = {};
    uint64_t global_memery_cachesize_ = 0;
    uint32_t compute_units_ = 0;
    uint32_t max_freq_ = 0;
    uint64_t local_memory_size_ = 0;
    std::string default_build_opts_ = "";
    GpuInfo gpu_info_;
    bool support_fp16_ = false;
    bool fp16_enable_ = false;
    Precision precision_ = PRECISION_AUTO;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_OPENCL_OPENCL_RUNTIME_H_
