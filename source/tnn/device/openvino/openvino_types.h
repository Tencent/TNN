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

#ifndef TNN_SOURCE_TNN_DEVICE_OPENVINO_OPENVINO_TYPES_H_
#define TNN_SOURCE_TNN_DEVICE_OPENVINO_OPENVINO_TYPES_H_

#include <cstdint>
#include <map>
#include <string>

#include <ngraph/node.hpp>
#include <ngraph/op/op.hpp>

#include "tnn/core/common.h"
#include "tnn/core/macro.h"
#include "tnn/core/status.h"
#include "tnn/extern_wrapper/foreign_tensor.h"

namespace TNN_NS {

//@brief Empty op 
class EmptyNode: public ngraph::Node {
    using NodeTypeInfo = Node::type_info_t;
public:
    static NodeTypeInfo type_info;
    const NodeTypeInfo& get_type_info() const override { return type_info; }
};

//@brief Base Type of a OpenvinoTensor 
class OpenvinoTensor: public ForeignTensor {
public:
    explicit OpenvinoTensor();

    // @brief virtual destructor
    virtual ~OpenvinoTensor();

    //@brief get the ForeignTensor
    std::shared_ptr<ngraph::Node> GetNode();

    //@brief set the ForeignTensor
    Status SetNode(std::shared_ptr<ngraph::Node> node);

protected:
    std::shared_ptr<ngraph::Node> node_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_OPENVINO_OPENVINO_TYPES_H_
