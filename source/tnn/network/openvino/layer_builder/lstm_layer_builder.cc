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

#include <cmath>
#include <memory>

#include <inference_engine.hpp>
#include <ngraph/ngraph.hpp>
#include <ngraph/node.hpp>
#include <ngraph/op/op.hpp>
#include <ngraph/opsets/opset.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "tnn/extern_wrapper/foreign_blob.h"
#include "tnn/extern_wrapper/foreign_tensor.h"
#include "tnn/layer/base_layer.h"
#include "tnn/network/openvino/layer_builder/openvino_layer_builder.h"
#include "tnn/network/openvino/openvino_types.h"
#include "tnn/utils/dims_vector_utils.h"

#include "tnn/network/openvino/custom_layer/custom_lstm_onnx.h"

namespace TNN_NS {

DECLARE_OPENVINO_LAYER_BUILDER(LSTMONNX, LAYER_LSTMONNX);

Status LSTMONNXOVLayerBuilder::Build() {
    auto paramlist = dynamic_cast<LSTMONNXLayerParam *>(param_);

    if (GetInputNodes().size() <= 0) {
        LOGE("Error: 0 input nodes\n");
        return TNNERR_INIT_LAYER;
    }
    auto input_nodes = GetInputNodes();

    // // input0: X:[sequence,batch,inputsize] -> [batch,sequence,inputsize] permute node (1,0,2)
    // // input4: hidden initial state:[direction,batch,hiddensize] -> [batch,direction,hiddensize] permute node (1,0,2)
    // // input5: hidden initial state:[direction,batch,hiddensize] -> [batch,direction,hiddensize] permute node (1,0,2)
    // // input1: W: const node
    // // input2: R: const node
    // // input3: bias:[direction, 8*hiddensize] -> [direction, 4*hiddensize] slice node, add node
    // // output: Y [batch,sequence,outputsize] -> [sequence,batch,outputsize] permute node[1,0,2]

    // auto directions = paramlist->direction >= 2 ? ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL
    //                                             : ngraph::op::RecurrentSequenceDirection::FORWARD;
    // const std::vector<float> activations_alpha = {};
    // const std::vector<float> activations_beta  = {};
    // const std::vector<std::string> activations = {"sigmoid", "sigmoid", "sigmoid"};
    // std::vector<int> permutePattern            = {1, 0, 2};
    // std::vector<int> sliceAxis                 = {1};
    // std::vector<int> sequence                  = {(int)input_nodes[0]->get_output_shape(0)[0]};
    // std::vector<int> lstm_permute              = {2, 0, 1, 3};
    // std::vector<int> lstm_reshape              = {0, 0, -1};

    // auto orderNode =
    //     std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::i32, ngraph::Shape{3}, permutePattern);
    // auto splitNode = std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::i32, ngraph::Shape{}, sliceAxis);
    // auto sequenceNode = ngraph::op::Constant::create(ngraph::element::i32, ngraph::Shape{1}, sequence);
    // auto reshapeNode =
    //     std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::i32, ngraph::Shape{3}, lstm_reshape);
    // // auto zeroNode = ngraph::op::Constant::create(
    // //                           ngraph::element::f32,
    // //                           ngraph::Shape{(paramlist->direction >= 2 ? 2UL : 1UL),
    // //                                 3UL * static_cast<size_t>(paramlist->hidden_size)},
    // //                           std::vector<float>{0.f});

    // auto inputNode       = std::make_shared<ngraph::op::Transpose>(input_nodes[0]->output(0), orderNode);
    // auto hiddenStateNode = std::make_shared<ngraph::op::Transpose>(input_nodes[4]->output(0), orderNode);
    // auto cellStateNode   = std::make_shared<ngraph::op::Transpose>(input_nodes[5]->output(0), orderNode);
    // auto sliceNode       = std::make_shared<ngraph::op::v1::Split>(input_nodes[3]->output(0), splitNode, 2);
    // auto addNode         = std::make_shared<ngraph::op::v0::Add>(sliceNode->output(0), sliceNode->output(1));
    // auto lstmNode        = std::make_shared<ngraph::op::v5::LSTMSequence>(inputNode->output(0), //input
    //                                                                       hiddenStateNode->output(0), //hiddenstate
    //                                                                       cellStateNode->output(0), //cellstate
    //                                                                       sequenceNode->output(0),
    //                                                                       input_nodes[1]->output(0), //W
    //                                                                       input_nodes[2]->output(0), //R
    //                                                                       addNode->output(0), //B
    //                                                                       paramlist->hidden_size,
    //                                                                       directions,
    //                                                                       activations_alpha,
    //                                                                       activations_beta,
    //                                                                       activations);
    
    // auto lstmOrderNode =
    //     std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::i32, ngraph::Shape{4}, lstm_permute);
    // auto lstmPermuteNode = std::make_shared<ngraph::op::Transpose>(lstmNode->output(0), lstmOrderNode);
    // auto lstmReshapeNode = std::make_shared<ngraph::op::v1::Reshape>(lstmPermuteNode->output(0), reshapeNode, true);

    // auto lstmHiddenPermuteNode = std::make_shared<ngraph::op::Transpose>(lstmNode->output(1), orderNode);
    // auto lstmCellPermuteNode   = std::make_shared<ngraph::op::Transpose>(lstmNode->output(2), orderNode);

    // lstmReshapeNode->validate_and_infer_types();
    // lstmReshapeNode->set_friendly_name(paramlist->name);

    // ngraph::NodeVector outputNodes;
    // outputNodes.push_back(lstmReshapeNode);
    // outputNodes.push_back(lstmHiddenPermuteNode);
    // outputNodes.push_back(lstmCellPermuteNode);
    // SetOutputTensors(outputNodes);

    // return TNN_OK;



    // build to openvino lstm sequence node success, but crc wrong, use custom lstm op instead

    ngraph::OutputVector inputs;
    for (auto item : input_nodes) {
        inputs.push_back(item->output(0));
    }
    auto lstmNode = std::make_shared<CustomLSTMONNXOp>(
        inputs, base_layer_, GetInputBlobs(), GetOutputBlobs());

    lstmNode->validate_and_infer_types();
    lstmNode->set_friendly_name(paramlist->name);

    ngraph::NodeVector outputNodes;
    for (auto &iter : lstmNode->outputs()) {
        outputNodes.push_back(iter.get_node_shared_ptr());
    }
    SetOutputTensors(outputNodes);

    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(LSTMONNX, LAYER_LSTMONNX);
REGISTER_CUSTOM_TYPE(LAYER_LSTMONNX);
}  // namespace TNN_NS