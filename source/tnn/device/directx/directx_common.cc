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

#include "tnn/device/directx/directx_common.h"

#include <memory>

#define NOMINMAX
#include <d3dcommon.h>
#include <d3d11.h>
#undef LoadLibrary

#include "tnn/core/status.h"
#include "tnn/device/directx/directx_util.h"
#include "tnn/device/directx/directx_device.h"

namespace TNN_NS {

namespace directx {

DirectXProfilingData::DirectXProfilingData() {
}

Status DirectXProfilingData::Init() {
    std::shared_ptr<ID3D11Device> device(nullptr);
    RETURN_ON_NEQ(GetID3DDevice(device), TNN_OK);

    D3D11_QUERY_DESC desc;
    desc.Query = D3D11_QUERY_TIMESTAMP;
    desc.MiscFlags = 0u;

    if (start_point == nullptr) {
        HRESULT hr = device->CreateQuery(&desc, &start_point);
        if (FAILED(hr)) {
            start_point = nullptr;
            LOGE("Create ID3DQuery failed %lu\n", hr);
            return Status(TNNERR_DX_RESOURCE_CREATION, "Create ID3DQuery failed");
        }
    }

    if (end_point == nullptr) {
        HRESULT hr = device->CreateQuery(&desc, &end_point);
        if (FAILED(hr)) {
            end_point = nullptr;
            LOGE("Create ID3DQuery failed %lu\n", hr);
            return Status(TNNERR_DX_RESOURCE_CREATION, "Create ID3DQuery failed");
        }
    }

    return TNN_OK;
}

void DirectXProfilingData::Begin() {
    std::shared_ptr<ID3D11DeviceContext> context(nullptr);
    GetID3DContext(context);

    if (start_point && context) {
        context->End(start_point);
    }
}

void DirectXProfilingData::End() {
    std::shared_ptr<ID3D11DeviceContext> context(nullptr);
    GetID3DContext(context);

    if (end_point && context) {
        context->End(end_point);
    }
}

uint64_t DirectXProfilingData::Finalize() {
    std::shared_ptr<ID3D11DeviceContext> context(nullptr);
    GetID3DContext(context);

    uint64_t diff = 0;
    if (start_point && end_point && context ) {
        uint64_t start_time, end_time;
        HRESULT hr_start = context->GetData(start_point, &start_time, sizeof(uint64_t), 0);
        HRESULT hr_end   = context->GetData(end_point, &end_time, sizeof(uint64_t), 0);
        if (FAILED(hr_start) || FAILED(hr_end)) {
            LOGE("ID3DContext Get Data failed, ret: 0x%lu, 0x%lu\n", hr_start, hr_end);
            return diff;
        }
        diff = end_time - start_time;
    }

    LOGI("diff is %lu\n", diff);

    return diff;
}


DirectXProfilingData::~DirectXProfilingData() {
    if (start_point != nullptr) {
        start_point->Release();
    }
    if (end_point != nullptr) {
        end_point->Release();
    }
}

#if TNN_PROFILE
DirectXProfilingResult::~DirectXProfilingResult() {
    if (disjoint_ != nullptr) {
        disjoint_->Release();
    }
}

Status DirectXProfilingResult::Init() {
    std::shared_ptr<ID3D11Device> device(nullptr);
    RETURN_ON_NEQ(GetID3DDevice(device), TNN_OK);

    D3D11_QUERY_DESC desc;
    desc.Query = D3D11_QUERY_TIMESTAMP_DISJOINT;
    desc.MiscFlags = 0u;

    if (disjoint_== nullptr) {
        HRESULT hr = device->CreateQuery(&desc, &disjoint_);
        if (FAILED(hr)) {
            disjoint_ = nullptr;
            LOGE("Create ID3DQuery of Disjoint failed %lu\n", hr);
            return Status(TNNERR_DX_RESOURCE_CREATION, "Create ID3DQuery failed");
        }
    }

    return TNN_OK;
}

void DirectXProfilingResult::Begin() {
    std::shared_ptr<ID3D11DeviceContext> context(nullptr);
    GetID3DContext(context);

    if (disjoint_ && context) {
        context->Begin(disjoint_);
    }
}

void DirectXProfilingResult::End() {
    std::shared_ptr<ID3D11DeviceContext> context(nullptr);
    GetID3DContext(context);

    if (disjoint_ && context) {
        context->End(disjoint_);
    }
}


std::string DirectXProfilingResult::GetProfilingDataInfo() {

    RETURN_VALUE_ON_NEQ(GetD3DQueryData(), TNN_OK, "Profiling got no valid data!\n");

    return ProfileResult::GetProfilingDataInfo();
}

Status DirectXProfilingResult::GetD3DQueryData() {
    std::shared_ptr<ID3D11DeviceContext> context(nullptr);
    RETURN_ON_NEQ(GetID3DContext(context), TNN_OK);

    // wait until the execution finished.
    while (context->GetData(disjoint_, NULL, 0, 0) == S_FALSE)
    {
        // Wait 1 millisec, give other threads a chance to run
        Sleep(1);       
    }

    D3D10_QUERY_DATA_TIMESTAMP_DISJOINT disjoint_value;
    context->GetData(disjoint_, &disjoint_value, sizeof(disjoint_value), 0);
    if (disjoint_value.Disjoint) {
        LOGE("Profiling data not valid due to disjoint.\n");
        return Status(TNNERR_DX_PROFILING, "Profiling data not valid due to disjoint.\n");
    }

    for(auto data : profiling_data_) {
        auto dx_data = std::dynamic_pointer_cast<DirectXProfilingData>(data);
        if (dx_data) {
            // milliseconds
            dx_data->kernel_time =  float(dx_data->Finalize()) / float(disjoint_value.Frequency) * 1000.0f;
        }
    }

    return TNN_OK;
}

#endif

} // namespace directx

}  // namespace TNN_NS
