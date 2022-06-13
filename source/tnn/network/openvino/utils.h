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

#ifndef TNN_SOURCE_TNN_NETWORK_OPENVINO_UTILS_H_
#define TNN_SOURCE_TNN_NETWORK_OPENVINO_UTILS_H_

#include <ie_precision.hpp>
#include <ngraph/ngraph.hpp>
#include <ngraph/node.hpp>
#include <ngraph/op/op.hpp>
#include <ngraph/opsets/opset.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "tnn/core/common.h"
#include "tnn/core/macro.h"
#include "tnn/interpreter/raw_buffer.h"

namespace TNN_NS {

#define ADD_CUSTOM_NODE(type, name)                                                                                    \
    std::shared_ptr<ngraph::Node> customNode;                                                                          \
    auto input_nodes = GetInputNodes();                                                                                \
    ngraph::OutputVector inputs;                                                                                       \
    for (auto item : input_nodes) {                                                                                    \
        inputs.push_back(item->output(0));                                                                             \
    }                                                                                                                  \
    customNode = std::make_shared<Custom##type##Op>(inputs, base_layer_, GetInputBlobs(), GetOutputBlobs());           \
    customNode->set_friendly_name(name);                                                                               \
    customNode->validate_and_infer_types();                                                                            \
    ngraph::NodeVector outputNodes;                                                                                    \
    outputNodes.push_back(customNode);                                                                                 \
    SetOutputTensors(outputNodes);

ngraph::element::Type_t ConvertToOVDataType(DataType type);
std::shared_ptr<ngraph::op::Constant> ConvertToConstNode(RawBuffer *buffer);
DataType ConvertOVPrecisionToDataType(const InferenceEngine::Precision &precision);
InferenceEngine::Precision ConvertOVTypeToPrecision(ngraph::element::Type_t type);

}  //  namespace TNN_NS

#endif  //  TNN_SOURCE_TNN_NETWORK_TENSORRT_UTILS_H_
