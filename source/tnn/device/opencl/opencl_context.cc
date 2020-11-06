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

#include "tnn/device/opencl/opencl_context.h"
#include "tnn/core/profile.h"
#include "tnn/device/opencl/opencl_utils.h"
#include "tnn/utils/string_format.h"

#include "sys/time.h"

namespace TNN_NS {

OpenCLContext::OpenCLContext() : Context() {
    // Get OpenCL Runtime
    opencl_runtime_ = OpenCLRuntime::GetInstance();
    OpenCLRuntime::IncreaseRef();
}

OpenCLContext::~OpenCLContext() {
    command_queue_.reset();
    OpenCLRuntime::DecreaseRef();
}

Status OpenCLContext::GetCommandQueue(void** command_queue) {
    *command_queue = command_queue_.get();
    return TNN_OK;
}

cl::CommandQueue* OpenCLContext::CommandQueue() {
    return command_queue_.get();
}

#if TNN_PROFILE
void OpenCLContext::StartProfile() {
    Context::StartProfile();
    profiling_result_ = std::make_shared<OpenCLProfileResult>();
}

OpenCLProfilingData::~OpenCLProfilingData() {}

OpenCLProfileResult::~OpenCLProfileResult() {}

std::string OpenCLProfileResult::GetProfilingDataInfo() {
    // show the time cost of each layer
    std::string title                     = "Profiling Data";
    const std::vector<std::string> header = {"name",       "Op Type",   "Kernel(ms)",  "Queued(ms)",   "Submit(ms)",
                                             "Start(ms)",  "End(ms)",   "Input(NCHW)", "Output(NCHW)", "Filter(OIHW)",
                                             "Stride",     "Pad",       "Dilation",    "GFlops",       "BW(GB/s)",
                                             "GWS(0,1,2)", "LWS(0,1,2)"};

    std::vector<std::vector<std::string>> data;

    double kernel_time_sum = 0;
    for (auto item : profiling_data_) {
        OpenCLProfilingData* p = dynamic_cast<OpenCLProfilingData*>(item.get());
        if (nullptr == p) {
            LOGE("OpenCLProfilingData is nil\n");
            return "";
        }
        // GetProfiling
        GetProfilingTime(&p->event, p->kernel_time, p->event_queued, p->event_submit, p->event_start, p->event_end);
    }
    auto p = dynamic_cast<OpenCLProfilingData*>(profiling_data_[0].get());
    if (nullptr == p) {
        LOGE("OpenCLProfilingData is nil\n");
        return "";
    }
    double profile_start = p->event_queued;
    for (auto item : profiling_data_) {
        OpenCLProfilingData* p = dynamic_cast<OpenCLProfilingData*>(item.get());
        if (nullptr == p) {
            LOGE("OpenCLProfilingData is nil\n");
            return "";
        }

        p->event_queued = (p->event_queued - profile_start) / 1000000.0;
        p->event_submit = (p->event_submit - profile_start) / 1000000.0;
        p->event_start  = (p->event_start - profile_start) / 1000000.0;
        p->event_end    = (p->event_end - profile_start) / 1000000.0;
    }

    for (auto item : profiling_data_) {
        OpenCLProfilingData* p = dynamic_cast<OpenCLProfilingData*>(item.get());
        if (nullptr == p) {
            LOGE("OpenCLProfilingData is nil\n");
            return "";
        }
        std::vector<std::string> tuples = {};
        tuples.reserve(32);

        tuples.push_back(p->layer_name);
        tuples.push_back(p->op_name);
        tuples.push_back(DoubleToString(p->kernel_time));
        tuples.push_back(DoubleToString(p->event_queued));
        tuples.push_back(DoubleToString(p->event_submit));
        tuples.push_back(DoubleToString(p->event_start));
        tuples.push_back(DoubleToString(p->event_end));
        tuples.push_back(VectorToString(p->input_dims));
        tuples.push_back(VectorToString(p->output_dims));
        tuples.push_back(VectorToString(p->kernel_shape));
        tuples.push_back(VectorToString(p->stride_shape));
        tuples.push_back(VectorToString(p->pad_shape));
        tuples.push_back(VectorToString(p->dilation_shape));
        tuples.push_back(DoubleToStringFilter(p->flops / p->kernel_time));
        tuples.push_back(DoubleToStringFilter(p->bandwidth / p->kernel_time));
        tuples.push_back(VectorToString(p->global_worksize));
        tuples.push_back(VectorToString(p->local_worksize));

        data.emplace_back(tuples);
        kernel_time_sum += p->kernel_time;
    }

    std::string detailed_string = StringFormatter::Table(title, header, data);

    std::string summary_string = GetProfilingDataSummary(false);

    std::ostringstream ostr;
    ostr << "kernel runtime total: " << kernel_time_sum << " ms\n\n";

    return detailed_string + summary_string + ostr.str();
}
#endif

// external dependent library load
Status OpenCLContext::LoadLibrary(std::vector<std::string> path) {
    return Init();
}

Status OpenCLContext::OnInstanceForwardBegin() {
    return Context::OnInstanceForwardBegin();
}

Status OpenCLContext::OnInstanceForwardEnd() {
    return TNN_OK;
}

// synchronize will wait until the comman queue finish
Status OpenCLContext::Synchronize() {
    cl_int result = command_queue_->finish();
    if (result == 0) {
        return TNN_OK;
    } else {
        return Status(TNNERR_OPENCL_FINISH_ERROR, "command queue finish falied");
    }
}

// opencl kernel flush strategy, some devices(special for huawei device) whave serious latency.
unsigned int OpenCLContext::AddAndGetFlushCount() {
    flush_count_++;
    return flush_count_;
}

// Init Will create command queue, get fp16 info
Status OpenCLContext::Init() {
    if (opencl_runtime_ == nullptr) {
        return Status(TNNERR_OPENCL_RUNTIME_ERROR, "opencl_runtime is nullptr");
    }

    Status status = opencl_runtime_->Init();
    if (status != TNN_OK) {
        LOGE("OpenCL Runtime Init() failed (ret = %d)!\n", (int)status);
        return status;
    }

    cl_command_queue_properties properties = 0;
#if TNN_PROFILE
    properties |= CL_QUEUE_PROFILING_ENABLE;
#endif

    cl_int err;
    command_queue_ =
        std::make_shared<cl::CommandQueue>(*opencl_runtime_->Context(), *opencl_runtime_->Device(), properties, &err);
    if (err != CL_SUCCESS) {
        LOGE("Command Queue create failed! (ERROR CODE: %d)\n", err);
        return Status(TNNERR_DEVICE_CONTEXT_CREATE, "Command Queue create failed!");
    }

    bool ret = opencl_runtime_->SetPrecision(precision_);
    if (ret) {
        LOGE("opencl set precision %d\n", precision_);
    } else {
        LOGE("opencl set fp16 precision failed, precision set: %d\n", opencl_runtime_->GetPrecision());
    }

    return TNN_OK;
}

}  // namespace TNN_NS
