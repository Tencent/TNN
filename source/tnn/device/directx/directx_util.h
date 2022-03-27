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

#ifndef TNN_DIRECTX_UTIL_H_
#define TNN_DIRECTX_UTIL_H_

#include <string.h>
#include <cstdlib>

#if TNN_PROFILE
#include <chrono>
#endif

#define NOMINMAX
#include <d3dcommon.h>
#include <d3d11.h>
#undef LoadLibrary

#include "tnn/core/blob.h"
#include "tnn/core/macro.h"
#include "tnn/device/directx/directx_memory.h"
#include "tnn/memory_manager/blob_memory_size_info.h"

namespace TNN_NS {
namespace directx {
#if TNN_PROFILE
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::time_point;
using std::chrono::system_clock;
struct Timer {
public:
    void Start() {
        start_ = system_clock::now();
    }
    float TimeEclapsed() {
        stop_ = system_clock::now();
        float elapsed = duration_cast<microseconds>(stop_ - start_).count() / 1000.0f;
        start_ = system_clock::now();
        return elapsed;
    }
private:
    time_point<system_clock> start_;
    time_point<system_clock> stop_;
};
#endif

struct ParamCB
    {
    float scale0;
    float scale1;
    float scale2;
    float scale3;

    float bias0;
    float bias1;
    float bias2;
    float bias3;

    int n;
    int c;
    int h;
    int w;
    };

// Tell the memory type from a blob Description sruct
DirectXMemoryType GetMemoryType(BlobDesc desc);

// Tell the memory type from a blob memory size info, which is used by the AbstactDevice::Allocate function
DirectXMemoryType GetMemoryType(BlobMemorySizeInfo size_info);

Status DispatchShader(ID3D11ComputeShader* cs,
                      std::vector<std::shared_ptr<ID3D11ShaderResourceView>> srvs,
                      std::vector<std::shared_ptr<ID3D11UnorderedAccessView>> uavs,
                      std::vector<ID3D11Buffer*> InputCBBuffer_ptrs,
                      std::vector<unsigned int> grid);

Status AllocateBuffer(std::shared_ptr<DirectXMemory> buffer_out,
                      BlobMemorySizeInfo& desc,
                      const void * inital_data);

Status AllocateConstantBuffer(ID3D11Buffer* &pInputCBBuffer,
                              ParamCB &paramCB_data);

}
}  // namespace TNN_NS

#endif
