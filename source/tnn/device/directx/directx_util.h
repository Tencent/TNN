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
#include <directxmath.h>

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

struct ParamCB {
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

Status DispatchShader(const std::shared_ptr<ID3D11ComputeShader> cs, 
                      const std::vector<std::shared_ptr<ID3D11ShaderResourceView>> srvs,  
                      const std::vector<std::shared_ptr<ID3D11UnorderedAccessView>> uavx,  
                      const std::vector<ID3D11Buffer*> const_bufs,  
                      const std::vector<int> grid);

Status GetShaderByName(const std::string, std::shared_ptr<ID3D11ComputeShader> &shader );

Status GetID3DDevice(std::shared_ptr<ID3D11Device> &device);

Status GetID3DContext(std::shared_ptr<ID3D11DeviceContext> &context);

Status UpdateTexture2D(void*data_ptr,
                       std::vector<int> dims,
                       std::shared_ptr<DirectXMemory> &texture);

Status UpdateConv2DFilterTexture2D(void* data_ptr,
                                   std::vector<int> dims,
                                   int image_width,
                                   int image_height,
                                   std::shared_ptr<DirectXMemory> &texture_memory);

Status UpdateConvDWFilterTexture2D(void* data_ptr,
                                   std::vector<int> dims,
                                   int image_width,
                                   int image_height,
                                   std::shared_ptr<DirectXMemory> &texture_memory);

template<typename T>
Status CreateConstBuffer(const T &host_value, 
                        std::shared_ptr<ID3D11Device> device, 
                        std::shared_ptr<ID3D11Buffer> &buf) {

    D3D11_SUBRESOURCE_DATA init_data;
    init_data.pSysMem = &host_value;
    init_data.SysMemPitch = 0;
    init_data.SysMemSlicePitch = 0;

    D3D11_BUFFER_DESC constant_buffer_desc = {};
    constant_buffer_desc.ByteWidth = ROUND_UP(sizeof(T), 16);
    constant_buffer_desc.Usage = D3D11_USAGE_DEFAULT;
    constant_buffer_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    constant_buffer_desc.CPUAccessFlags = 0;

    LOGD("const buffer size_in_bytes:%lu\n", constant_buffer_desc.ByteWidth);
    if (constant_buffer_desc.ByteWidth >= D3D11_REQ_CONSTANT_BUFFER_ELEMENT_COUNT ) {
        LOGE("too large const buffer, size_in_bytes:%lu should less than %lu\n", constant_buffer_desc.ByteWidth, D3D11_REQ_CONSTANT_BUFFER_ELEMENT_COUNT);
        return Status(TNNERR_DX_BUFFER_ALOCATE_ERR, "too large const buffer");
    }

    ID3D11Buffer * p_d3d_const_buffer;
    HRESULT hr = device->CreateBuffer( &constant_buffer_desc, &init_data, &p_d3d_const_buffer);
    if (FAILED(hr)) {
        LOGE("Create const buffer failed, err code:0x%X", hr);
        return Status(TNNERR_DX_BUFFER_ALOCATE_ERR, "Create const buffer failed");
    }

    /*
    D3D11_MAPPED_SUBRESOURCE maped_res;
    pd3dImmediateContext->Map(p_d3d_const_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &maped_res);
    memcpy( maped_res.pData, &host_value, sizeof(T));
    pd3dImmediateContext->Unmap( p_d3d_const_buffer, 0 );
    */

    buf = std::shared_ptr<ID3D11Buffer>(p_d3d_const_buffer, [](ID3D11Buffer* p) {p->Release();} );

    return TNN_OK;
}

enum GroupWeightsFormat { GOIHW, GIOHW };

template <typename T, typename Dim>
inline void GROUP_PADDING(const T *src, T *dst, Dim G, Dim O, Dim I, Dim H, Dim W, GroupWeightsFormat src_format) {
    int input_channel_per_group = I / G;
    int group_size_in_o         = O / G;

    for (Dim o = 0; o < O; o++) {
        for (Dim i = 0; i < I; i++) {
            for (Dim h = 0; h < H; h++) {
                for (Dim w = 0; w < W; w++) {
                    int dst_idx = o * I * H * W + i * H * W + h * W + w;

                    int group_id  = o / group_size_in_o;
                    int valid_i_b = group_id * input_channel_per_group;
                    int valid_i_e = valid_i_b + input_channel_per_group;
                    if (i < valid_i_b || i >= valid_i_e) {
                        dst[dst_idx] = 0;
                    } else {
                        int g_idx = group_id;
                        int o_idx = o % group_size_in_o;
                        int i_idx = i % input_channel_per_group;
                        int h_idx = h;
                        int w_idx = w;
                        // src is GOIHW
                        int src_idx;
                        if (src_format == GOIHW) {
                            src_idx = g_idx * group_size_in_o * input_channel_per_group * H * W +
                                      o_idx * input_channel_per_group * H * W + i_idx * H * W + h_idx * W + w_idx;
                        } else {
                            // src is GIOHW
                            src_idx = g_idx * input_channel_per_group * group_size_in_o * H * W +
                                      i_idx * group_size_in_o * H * W + o_idx * H * W + h_idx * W + w_idx;
                        }
                        dst[dst_idx] = src[src_idx];
                    }
                }
            }
        }
    }
}

Status AllocateBuffer(std::shared_ptr<DirectXMemory> buffer_out,
                      BlobMemorySizeInfo& desc,
                      const void * inital_data);

Status AllocateConstantBuffer(ID3D11Buffer* &pInputCBBuffer,
                              ParamCB &paramCB_data);

}
}  // namespace TNN_NS

#endif
