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

#ifndef TNN_SOURCE_TNN_CORE_CONTEXT_H_
#define TNN_SOURCE_TNN_CORE_CONTEXT_H_

#include <memory>
#include <string>
#include <vector>

#include "tnn/core/status.h"
#include "tnn/core/profile.h"
#include "tnn/core/common.h"

namespace TNN_NS {

class Context {
public:
    // @brief virtual destructor
    virtual ~Context() {}

    // @brief load library
    virtual Status LoadLibrary(std::vector<std::string> path) = 0;

    // @brief get tnn command queue
    // @param command_queue device command queue for forward
    virtual Status GetCommandQueue(void** command_queue) = 0;

    // @brief share tnn command queue to another context
    virtual Status ShareCommandQueue(Context* context);
    
    // @brief before instace forword
    virtual Status OnInstanceForwardBegin();

    // @brief after instace forword
    virtual Status OnInstanceForwardEnd() = 0;

    // @brief before instance Reshape
    virtual Status OnInstanceReshapeBegin();

    // @brief after instace Reshape
    virtual Status OnInstanceReshapeEnd();

    // @brief wait for jobs in the current context to complete
    virtual Status Synchronize() = 0;

    // @brief set threads run on device
    virtual Status SetNumThreads(int num_threads);

    void SetPrecision(Precision precision);

    Precision GetPrecision();

    void SetEnableTuneKernel(bool enalbe_tune_kernel);

    bool GetEnableTuneKernel();

#if TNN_PROFILE
public:
    virtual void StartProfile();
    virtual std::shared_ptr<ProfileResult> FinishProfile();
    void AddProfilingData(std::shared_ptr<ProfilingData> pdata);

    bool profile_layer = false;

protected:
    std::shared_ptr<ProfileResult> profiling_result_ = nullptr;
#endif

protected:
    Precision precision_ = PRECISION_AUTO;
    bool enable_tune_kernel_ = true;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_CORE_CONTEXT_H_
