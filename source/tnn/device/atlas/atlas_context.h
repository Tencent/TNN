// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef TNN_SOURCE_TNN_DEVICE_ATLAS_ATLAS_CONTEXT_H_
#define TNN_SOURCE_TNN_DEVICE_ATLAS_ATLAS_CONTEXT_H_

#include "tnn/core/context.h"
#include "tnn/device/atlas/atlas_common_types.h"
#include "tnn/interpreter/raw_buffer.h"

namespace TNN_NS {

class AtlasContext : public Context {
public:
    // @brief deconstructor
    ~AtlasContext();

    // @brief setup with specified device id
    Status Setup(int device_id);

    // @brief load library
    virtual Status LoadLibrary(std::vector<std::string> path) override;

    // @brief get tnn command queue
    // @param command_queue device command queue for forward
    virtual Status GetCommandQueue(void** command_queue) override;

    // @brief set tnn command queue
    // @param command_queue device command queue for forward
    virtual Status SetCommandQueue(void* command_queue) override;

    // @brief share tnn command queue to another context
    virtual Status ShareCommandQueue(Context* context);

    // @brief before instance forward
    virtual Status OnInstanceForwardBegin() override;

    // @brief after instance forward
    virtual Status OnInstanceForwardEnd() override;

    // @brief wait for jobs in the current context to complete
    virtual Status Synchronize() override;

    // @brief get Atlas stream
    aclrtStream& GetAclrtStream();

    // @brief set Atlas stream
    void SetAclrtStream(const aclrtStream& stream);

    // @brief create Atlas stream
    Status CreateAclrtStream();

    // @brief get ModelType
    ModelType& GetModelType();

    // @brief set ModelType
    void SetModelType(ModelType model_type);

private:
    ModelType model_type_;
    int device_id_ = INT_MAX;

    // ACL Runtime Related
    aclrtStream aclrt_stream_ = nullptr;
};

}  //  namespace TNN_NS;

#endif  //  TNN_SOURCE_TNN_DEVICE_ATLAS_ATLAS_CONTEXT_H_
