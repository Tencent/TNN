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

#ifndef TNN_SOURCE_TNN_DEVICE_CPU_CPU_CONTEXT_H_
#define TNN_SOURCE_TNN_DEVICE_CPU_CPU_CONTEXT_H_

#include "tnn/core/context.h"
#include "tnn/interpreter/raw_buffer.h"
namespace TNN_NS {

class ArmContext : public Context {
public:
    // load library
    virtual Status LoadLibrary(std::vector<std::string> path) override;

    // @brief get tnn command queue
    // @param command_queue device command queue for forward
    virtual Status GetCommandQueue(void** command_queue) override;
    
    // @brief share tnn command queue to another context
    virtual Status ShareCommandQueue(Context* context) override;
    
    // @brief befor instace forword
    virtual Status OnInstanceForwardBegin() override;

    // @brief after instace forword
    virtual Status OnInstanceForwardEnd() override;

    // @brief wait for jobs in the current context to complete
    virtual Status Synchronize() override;

    // @brief set threads run on device
    virtual Status SetNumThreads(int num_threads) override;

    // @brief get threads run on device
    virtual int GetNumThreads();

    void* GetSharedWorkSpace(size_t size);
    void* GetSharedWorkSpace(size_t size, int index);

private:
    int num_threads_ = 1;
    std::vector<RawBuffer> work_space_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_CPU_CPU_CONTEXT_H_
