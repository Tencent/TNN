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

#include "tnn/device/openvino/openvino_types.h"

#include <memory>

#include <ngraph/node.hpp>

#include "tnn/core/blob.h"
#include "tnn/extern_wrapper/foreign_tensor.h"

namespace TNN_NS {

EmptyNode::NodeTypeInfo EmptyNode::type_info{"EmptyNode", 0};

//@brief create OpenvinoTensor
OpenvinoTensor::OpenvinoTensor() {
    node_ = std::make_shared<EmptyNode>();
}

//@brief OpenvinoTensor destructor
OpenvinoTensor::~OpenvinoTensor() {
}

//@brief get the ForeignTensor
std::shared_ptr<ngraph::Node> OpenvinoTensor::GetNode() {
    return node_;
}

//@brief set the ForeignTensor
Status OpenvinoTensor::SetNode(std::shared_ptr<ngraph::Node> node) {
    node_ = node;
    return TNN_OK;
}


}  // namespace TNN_NS
