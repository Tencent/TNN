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

#ifndef TNN_SOURCE_TNN_DEVICE_OPENCL_OPENCL_CONTEXT_H_
#define TNN_SOURCE_TNN_DEVICE_OPENCL_OPENCL_CONTEXT_H_

#include <memory>

#include "tnn/core/context.h"
#include "tnn/core/profile.h"
#include "tnn/device/opencl/opencl_runtime.h"

namespace TNN_NS {

struct OpenCLProfilingData : public ProfilingData {
    virtual ~OpenCLProfilingData();
    cl::Event event;
    double event_queued;
    double event_submit;
    double event_start;
    double event_end;
    std::vector<uint32_t> global_worksize = {};
    std::vector<uint32_t> local_worksize = {};
};

#if TNN_PROFILE
class OpenCLProfileResult : public ProfileResult {
public:
    virtual ~OpenCLProfileResult();
    virtual std::string GetProfilingDataInfo() override;
};
#endif

class OpenCLContext : public Context {
public:
    OpenCLContext();
    ~OpenCLContext();

    // @brief get tnn command queue
    // @param command_queue device command queue for forward
    Status GetCommandQueue(void **command_queue) override;

    // @brief share tnn command queue to another context
    Status ShareCommandQueue(Context* context) override;

    /**
     * @brief get CommandQueue
     */
    cl::CommandQueue *CommandQueue();

    cl::CommandQueue *TuneCommandQueue();

    // load library
    virtual Status LoadLibrary(std::vector<std::string> path) override;
    /**
     * @brief befor instace forword
     * @param instance instace
     */
    virtual Status OnInstanceForwardBegin() override;
    /**
     * @brief after instace forword
     * @param instance instace
     */
    virtual Status OnInstanceForwardEnd() override;

     // @brief before instance Reshape
    virtual Status OnInstanceReshapeBegin() override;

    // @brief after instace Reshape
    virtual Status OnInstanceReshapeEnd() override;   

    // @brief wait for jobs in the current context to complete
    virtual Status Synchronize() override;

    // @brief add flush_count_ and return val
    unsigned int AddAndGetFlushCount();

#if TNN_PROFILE
public:
    virtual void StartProfile() override;
#endif

public:
    /**
     * @brief initialize opencl env
     */
    Status Init();

private:
    std::shared_ptr<cl::CommandQueue> command_queue_ = nullptr;
    std::shared_ptr<cl::CommandQueue> tune_command_queue_ = nullptr;
    std::shared_ptr<cl::CommandQueue> GetCommandQueue();
    OpenCLRuntime *opencl_runtime_ = nullptr;
    unsigned int flush_count_ = 0;
    cl_command_queue_properties properties_ = 0;

};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_OPENCL_OPENCL_CONTEXT_H_
