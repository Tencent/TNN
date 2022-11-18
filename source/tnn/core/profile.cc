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

#include "tnn/core/profile.h"
#include <time.h>
#include <iomanip>
#include <sstream>

#include "tnn/core/status.h"
#include "tnn/utils/string_format.h"

namespace TNN_NS {

ProfilingData::~ProfilingData() {}

bool ProfilingData::IsSameID(ProfilingData* data) {
    return data && op_name == data->op_name && layer_name == data->layer_name;
}

void ProfilingData::Add(ProfilingData* data) {
    if (!data || !IsSameID(data)) {
        return;
    }

    kernel_time += data->kernel_time;
    count += data->count;

    if (input_dims.size() <= 0) {
        input_dims = data->input_dims;
    }

    if (output_dims.size() <= 0) {
        output_dims = data->output_dims;
    }

    if (kernel_shape.size() <= 0) {
        kernel_shape = data->kernel_shape;
    }

    if (stride_shape.size() <= 0) {
        stride_shape = data->stride_shape;
    }

    if (pad_shape.size() <= 0) {
        pad_shape = data->pad_shape;
    }

    if (dilation_shape.size() <= 0) {
        dilation_shape = data->dilation_shape;
    }

    if (group <= 0) {
        group = data->group;
    }
}

#if TNN_PROFILE
ProfileResult::~ProfileResult() {}

void ProfileResult::Reset() {
    profiling_data_.clear();
}

/*
call this function in each layer
*/
void ProfileResult::AddProfilingData(std::shared_ptr<ProfilingData> pdata) {
    std::shared_ptr<ProfilingData> internal = nullptr;
    for (auto& item : profiling_data_) {
        if (item->IsSameID(pdata.get())) {
            internal = item;
            break;
        }
    }

    if (internal) {
        internal->Add(pdata.get());
    } else {
        profiling_data_.push_back(pdata);
    }
}

/*
call this function in network
*/
void ProfileResult::AddProfileResult(std::shared_ptr<ProfileResult> result) {
    auto result_profiling_data = result->GetData();
    for (auto pf_data : result_profiling_data) {
        AddProfilingData(pf_data);
    }
}

/*
get profilint data vector
*/
std::vector<std::shared_ptr<ProfilingData>> ProfileResult::GetData() {
    return profiling_data_;
}

std::string ProfileResult::GetProfilingDataTable(const std::string& title) {
    const std::vector<std::string> header = {"name",         "Op Type", "Kernel(ms)", "Input Dims", "Output Dims",
                                             "Filter(OIHW)", "Group", "Stride",  "Pad",        "Dilation"};
    std::vector<std::vector<std::string>> data;
    for (const auto& p : profiling_data_) {
        std::vector<std::string> tuple;
        tuple.reserve(16);

        tuple.push_back(p->layer_name);
        tuple.push_back(p->op_name);
        tuple.push_back(DoubleToString(p->kernel_time / p->count));
        tuple.push_back(VectorToString(p->input_dims));
        tuple.push_back(VectorToString(p->output_dims));
        tuple.push_back(VectorToString(p->kernel_shape));
        tuple.push_back(IntToStringFilter(p->group));
        tuple.push_back(VectorToString(p->stride_shape));
        tuple.push_back(VectorToString(p->pad_shape));
        tuple.push_back(VectorToString(p->dilation_shape));
        data.emplace_back(tuple);
    }
    std::string detailed_string = StringFormatter::Table(title, header, data);
    return detailed_string;
}
/*
format print profile info
*/
std::string ProfileResult::GetProfilingDataInfo() {
    std::string title        = "Profiling Data";
    std::string profile_data = GetProfilingDataTable(title);

    std::string sort_title = "Profiling Data Sort by Cost time";
    // descending sort profiling data by cost time
    auto compare = [](const std::shared_ptr<ProfilingData>& d1, const std::shared_ptr<ProfilingData>& d2) {
        return d1->kernel_time > d2->kernel_time;
    };
    std::sort(profiling_data_.begin(), profiling_data_.end(), compare);
    std::string sort_profile_data = GetProfilingDataTable(sort_title);

    std::string summary_string = GetProfilingDataSummary(true);

    double kernel_time_sum = 0;
    for (const auto& p : profiling_data_) {
        kernel_time_sum += p->kernel_time / p->count;
    }
    std::ostringstream ostr;
    ostr << "kernel runtime total: " << kernel_time_sum << " ms\n\n";

    return profile_data + sort_profile_data + summary_string + ostr.str();
}

std::string ProfileResult::GetProfilingDataSummary(bool do_average) {
    // show the time cost of each type layer
    std::string title_summary                     = "Summary";
    const std::vector<std::string> header_summary = {"Op Type", "Total Kernel Time(ms)", "Percent (%)"};

    double kernel_time_sum = 0;
    std::map<std::string, std::vector<float>> summary_map;
    for (auto p : profiling_data_) {
        if (do_average)
            kernel_time_sum += p->kernel_time / p->count;
        else
            kernel_time_sum += p->kernel_time;
        if (summary_map.find(p->op_name) == summary_map.end()) {
            std::vector<float> p_data;
            p_data.push_back(0.0f);
            summary_map[p->op_name] = p_data;
        }
    }
    for (auto p : profiling_data_) {
        if (summary_map.find(p->op_name) != summary_map.end()) {
            if (do_average)
                summary_map[p->op_name][0] += p->kernel_time / p->count;
            else
                summary_map[p->op_name][0] += p->kernel_time;
        }
    }
    auto summary_pair = SortMapByValue(summary_map);
    std::vector<std::vector<std::string>> data_summary;
    for (auto s : summary_pair) {
        std::vector<std::string> tuples;
        tuples.reserve(4);

        tuples.push_back(s.first);
        tuples.push_back(DoubleToString(s.second[0]));
        tuples.push_back(DoubleToString(s.second[0] / kernel_time_sum * 100));

        data_summary.emplace_back(tuples);
    }
    std::string show_string_summary = StringFormatter::Table(title_summary, header_summary, data_summary);
    return show_string_summary;
}
#endif

}  // namespace TNN_NS
