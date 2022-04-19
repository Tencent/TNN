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

#ifndef TNN_DIRECTX_COMMON_H_
#define TNN_DIRECTX_COMMON_H_

#include <memory>
#include <vector>
#include <unordered_set>
#include <chrono>

#define NOMINMAX
#include <d3dcommon.h>
#include <d3d11.h>
#undef LoadLibrary

#include "tnn/core/macro.h"
#include "tnn/core/status.h"
#include "tnn/device/directx/directx_macro.h"
#include "tnn/core/profile.h"

namespace TNN_NS {

namespace directx {

using std::chrono::time_point;
using std::chrono::system_clock;

class DxTimer {
public:
    DxTimer(std::string timer_info);
    void Start();
    void Stop();
    void Reset();
    void Print();

private:
    float min_;
    float max_;
    float sum_;
    std::string timer_info_;
    time_point<system_clock> start_;
    time_point<system_clock> stop_;
    int count_;
};

struct DirectXProfilingData : public ProfilingData {
    DirectXProfilingData();

    virtual ~DirectXProfilingData();

    // @brief Init the Query struct
    Status Init();

    // @brief Log the start time point 
    void Begin();

    // @brief Log the end time point 
    void End();

    // @brief Flush the query events
    void Flush();

    // @brief Reset count to zero
    void Reset();

    // @brief Update the kernel time to walltime by gpu frequency 
    void Finalize(double frequency);

    // @brief calc the kernel time, uint64_in d3d timestamp, not unix timestamp
    uint64_t GetLastTics();

    ID3D11Query * begin_query = nullptr;
    ID3D11Query * end_query = nullptr;

    bool need_flush = false;
    std::vector<uint64_t> kernel_tics;

    std::chrono::time_point<std::chrono::system_clock> begin_point;
    std::chrono::time_point<std::chrono::system_clock> end_point;

};

#if TNN_PROFILE
class DirectXProfilingResult :public ProfileResult {
public:

    virtual ~DirectXProfilingResult();

    // @brief Init the Query struct
    Status Init();

    // @brief begin the disjoint query 
    void Begin();

    // @brief end the disjoint query 
    void End();

    // @brief This function shows the detailed timing for each layer in the model.
    virtual std::string GetProfilingDataInfo();

    // @brief add profiling data of each layer
    virtual void AddProfilingData(std::shared_ptr<ProfilingData> pdata);

protected:

    // @brief Get timestamp from ID3D11Query struct after kernel execution
    Status UpdateDatasByFreqency();

    ID3D11Query * disjoint_ = nullptr;

    std::vector<std::shared_ptr<DirectXProfilingData>> data_list_;
    std::unordered_set<DirectXProfilingData *> data_set_;

};
#endif

} // namespace directx

} // namespace TNN_NS


#endif