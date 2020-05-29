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

#ifndef TNN_SOURCE_TNN_INTERPRETER_DEFAULT_MODEL_PACKER_H_
#define TNN_SOURCE_TNN_INTERPRETER_DEFAULT_MODEL_PACKER_H_

#include "tnn/core/status.h"
#include "tnn/interpreter/net_resource.h"
#include "tnn/interpreter/net_structure.h"

namespace TNN_NS {

// @brief DefaultModelPacker define common interface for rpn model, different
// interpreter different style model.
class DefaultModelPacker {
public:
    // @brief default constructor
    DefaultModelPacker(NetStructure *net_struct, NetResource *net_res);

    // @brief virtual destructor
    virtual ~DefaultModelPacker() = 0;

    // @brief save the rpn model into files
    virtual Status Pack(std::string proto_path, std::string model_path) = 0;

    //@brief GetNetStruture return network build info
    virtual NetStructure *GetNetStructure();

    //@brief GetNetResource return network weights data
    virtual NetResource *GetNetResource();

private:
    NetStructure *net_structure_;
    NetResource *net_resource_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_INTERPRETER_DEFAULT_MODEL_PACKER_H_
