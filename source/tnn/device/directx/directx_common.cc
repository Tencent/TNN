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
#include <chrono>

#define NOMINMAX
#include <d3dcommon.h>
#include <d3d11.h>
#undef LoadLibrary

#include "tnn/core/status.h"
#include "tnn/utils/string_format.h"
#include "tnn/device/directx/directx_util.h"
#include "tnn/device/directx/directx_device.h"

using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::time_point;
using std::chrono::system_clock;

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

    if (begin_query == nullptr) {
        HRESULT hr = device->CreateQuery(&desc, &begin_query);
        if (FAILED(hr)) {
            begin_query = nullptr;
            LOGE("Create ID3DQuery failed %lu\n", hr);
            return Status(TNNERR_DX_RESOURCE_CREATION, "Create ID3DQuery failed");
        }
    }

    if (end_query == nullptr) {
        HRESULT hr = device->CreateQuery(&desc, &end_query);
        if (FAILED(hr)) {
            end_query = nullptr;
            LOGE("Create ID3DQuery failed %lu\n", hr);
            return Status(TNNERR_DX_RESOURCE_CREATION, "Create ID3DQuery failed");
        }
    }

    return TNN_OK;
}

void DirectXProfilingData::Begin() {
    // flush first
    Flush();

    std::shared_ptr<ID3D11DeviceContext> context(nullptr);
    GetID3DContext(context);

    if (begin_query && context) {
        context->End(begin_query);
    }

    begin_point = system_clock::now();
}

void DirectXProfilingData::End() {
    std::shared_ptr<ID3D11DeviceContext> context(nullptr);
    GetID3DContext(context);

    if (end_query && context) {
        context->End(end_query);
    }

    end_point = system_clock::now();
    need_flush = true;

    // auto host_time = duration_cast<microseconds>(end_point - begin_point).count() / 1000.0f;
    // LOGI("%-10s %-10s host time:%.2fms\n", op_name.c_str(), layer_name.c_str(), host_time);
}

void DirectXProfilingData::Flush() {
    if (!need_flush) {
        return;
    }

    need_flush = false;
    auto tics = GetLastTics();
    if (tics < (0x1llu<<60)) {
        count++;
        submit_time += duration_cast<microseconds>(end_point - begin_point).count() / 1000.0f;
        kernel_tics.push_back(tics);
    }
}

void DirectXProfilingData::Reset() {
    need_flush = false;
    count = 0;
    submit_time = 0;
    kernel_tics.clear();
}

void DirectXProfilingData::Finalize(double frequency) {
    Flush();
    kernel_time = 0;
    for(auto v : kernel_tics) {
        // milliseconds
        auto vf =  double(v) / double(frequency) * 1000.0f;
        kernel_time += vf;
    }
}

uint64_t DirectXProfilingData::GetLastTics() {
    std::shared_ptr<ID3D11DeviceContext> context(nullptr);
    GetID3DContext(context);

    uint64_t diff = 0;
    if (begin_query && end_query && context ) {
        uint64_t start_time, end_time;
        HRESULT hr_start = context->GetData(begin_query, &start_time, sizeof(uint64_t), 0);
        HRESULT hr_end   = context->GetData(end_query, &end_time, sizeof(uint64_t), 0);
        if (FAILED(hr_start) || FAILED(hr_end)) {
            LOGE("ID3DContext Get Data failed, ret: 0x%lu, 0x%lu\n", hr_start, hr_end);
            return diff;
        }
        diff = end_time - start_time;
    }

    return diff;
}


DirectXProfilingData::~DirectXProfilingData() {
    if (begin_query != nullptr) {
        begin_query->Release();
    }
    if (end_query != nullptr) {
        end_query->Release();
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


    Status ret = UpdateDatasByFreqency();
    if (ret != TNN_OK) {
        LOGE("Profiling got invalid data!\n");
        return;
    }

    // Adding the queued ProfilingDatas to ProfilingResults::profiling_data_ now 
    // since it will aggregate the kernel::time, we don't get before
    for(auto data : data_list_) {
        ProfileResult::AddProfilingData(data);
    }
    data_list_.resize(0);
    data_set_.clear();
}

Status DirectXProfilingResult::UpdateDatasByFreqency() {
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

    for(auto data : data_list_) {
        data->Finalize(double(disjoint_value.Frequency));
    }

    return TNN_OK;
}

std::string DirectXProfilingResult::GetProfilingDataInfo() {
    // show the time cost of each layer
    std::string title                     = "Profiling Data";
    const std::vector<std::string> header = {"name",         "Op Type","Host time(ms)", "Kernel(ms)", "Input Dims", "Output Dims",
                                             "Filter(OIHW)", "Group", "Stride",  "Pad",        "Dilation"};

    std::vector<std::vector<std::string>> data;

    double kernel_time_sum = 0;

    for (auto p : profiling_data_) {
        std::vector<std::string> tuple;
        tuple.reserve(16);

        tuple.push_back(p->layer_name);
        tuple.push_back(p->op_name);
        tuple.push_back(DoubleToString(p->submit_time / p->count));
        tuple.push_back(DoubleToString(p->kernel_time / p->count));
        tuple.push_back(VectorToString(p->input_dims));
        tuple.push_back(VectorToString(p->output_dims));
        tuple.push_back(VectorToString(p->kernel_shape));
        tuple.push_back(IntToStringFilter(p->group));
        tuple.push_back(VectorToString(p->stride_shape));
        tuple.push_back(VectorToString(p->pad_shape));
        tuple.push_back(VectorToString(p->dilation_shape));

        data.emplace_back(tuple);

        kernel_time_sum += p->kernel_time / p->count;
    }

    std::string detailed_string = StringFormatter::Table(title, header, data);
    std::string summary_string  = GetProfilingDataSummary(true);

    std::ostringstream ostr;
    ostr << "kernel runtime total: " << kernel_time_sum << " ms\n\n";

    return detailed_string + summary_string + ostr.str();
}

void DirectXProfilingResult::AddProfilingData(std::shared_ptr<ProfilingData> pdata) {
    auto p = std::dynamic_pointer_cast<DirectXProfilingData>(pdata);
    if (p && data_set_.find(p.get()) == data_set_.end()) {
        p->Reset();
        data_list_.push_back(p);
        data_set_.insert(p.get());
    }
}

#endif

} // namespace directx

}  // namespace TNN_NS
