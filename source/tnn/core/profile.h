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

#ifndef TNN_INCLUDE_TNN_CORE_PROFILE_H_
#define TNN_INCLUDE_TNN_CORE_PROFILE_H_

#include <memory>
#include <string>
#include <vector>

#include "tnn/core/macro.h"

#pragma warning(push)
#pragma warning(disable : 4251)

namespace TNN_NS {

struct ProfilingData {
    virtual ~ProfilingData();
    /**layer name*/
    std::string layer_name = "";
    /**op type*/
    std::string op_name = "";
    /**kernel time*/
    double enqueue_time = 0;
    /**submit time*/
    double submit_time = 0;
    /**kernel time*/
    double kernel_time = 0;

    double flops     = 0;
    double bandwidth = 0;

    std::vector<int> input_dims     = {};
    std::vector<int> output_dims    = {};
    std::vector<int> kernel_shape   = {};
    std::vector<int> stride_shape   = {};
    std::vector<int> pad_shape      = {};
    std::vector<int> dilation_shape = {};
    int group                       = 0;

    int count = 1;

    void Add(ProfilingData *data);
    bool IsSameID(ProfilingData *data);
};

#if TNN_PROFILE
class ProfileResult {
public:
    virtual ~ProfileResult();

    // @brief reset for profile again
    void Reset();

    // @brief add profiling data of each layer
    void AddProfilingData(std::shared_ptr<ProfilingData> pdata);

    // @brief add profiling result
    void AddProfileResult(std::shared_ptr<ProfileResult> result);

    // @brief get profiling data
    virtual std::vector<std::shared_ptr<ProfilingData>> GetData();

    // @brief This function shows the detailed timing for each layer in the model.
    virtual std::string GetProfilingDataInfo();

    // @brief This function shows the detailed timing for each layer(sort by cost time) in the model.
    virtual std::string GetProfilingDataTable(const std::string& title);

protected:
    /*
     * This function shows an overview of the timings in the model.
     * the timing is grouped by the type of layer.
     */
    virtual std::string GetProfilingDataSummary(bool do_average);

    std::vector<std::shared_ptr<ProfilingData>> profiling_data_ = {};
};
#endif

}  // namespace TNN_NS

#pragma warning(pop)

#endif  // TNN_INCLUDE_TNN_CORE_PROFILE_H_
