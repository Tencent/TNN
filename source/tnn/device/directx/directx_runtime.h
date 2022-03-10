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

#ifndef TNN_SOURCE_TNN_DEVICE_DIRECTX_DIRECTX_RUNTIME_H_
#define TNN_SOURCE_TNN_DEVICE_DIRECTX_DIRECTX_RUNTIME_H_

#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>

#include <d3dcommon.h>
#include <d3d11.h>
#undef min
#undef max

#include "tnn/core/status.h"
#include "tnn/core/common.h"

namespace TNN_NS {

// enum GpuType { OTHER = 0, ADRENO = 1, MALI = 2, MALI_T = 3, MALI_G = 4, INTEL_GPU = 5, NVIDIA_GPU = 6};

// struct GpuInfo {
//     GpuType type = OTHER;
//     int model_num = 0;
//     float opencl_version = 0;
// };

// // Base GPU cache size used for computing local work group size.
// const int32_t g_base_gpu_mem_cachesize = 16384;

class DirectXRuntime {
public:
    static DirectXRuntime *GetInstance();

    ~DirectXRuntime();
    DirectXRuntime(const DirectXRuntime &) = delete;
    DirectXRuntime &operator=(const DirectXRuntime &) = delete;

    Status Init();

    std::shared_ptr<ID3D11Device> Device();
    std::shared_ptr<ID3D11DeviceContext> Context();

    // uint64_t DeviceGlobalMemeryCacheSize() const;
    // uint32_t DeviceComputeUnits() const;
    // uint32_t DeviceMaxFreq() const;
    // uint64_t DeviceLocalMemerySize() const;
    // uint64_t GetMaxWorkGroupSize(const cl::Kernel &kernel);
    // uint32_t GetSubGroupSize(const cl::Kernel &kernel, const cl::NDRange &range = cl::NullRange);
    // GpuInfo GetGpuInfo();

    std::vector<size_t> GetTexture2DMaxSize();

private:
    DirectXRuntime();

private:
    static std::shared_ptr<DirectXRuntime> g_singleton_;
    static std::mutex g_mutex_;

    std::shared_ptr<ID3D11Device> device_ = nullptr;
    std::shared_ptr<ID3D11DeviceContext> context_ = nullptr;

    std::vector<size_t> texture_2d_max_size_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_DIRECTX_DIRECTX_RUNTIME_H_
