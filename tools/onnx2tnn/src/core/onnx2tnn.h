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

#ifndef onnx2tnn_hpp
#define onnx2tnn_hpp

#include <float.h>
#include <limits.h>
#include <stdio.h>

#include <algorithm>
#include <fstream>
#include <string>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <exception>

#include "tnn/core/blob.h"
#include "tnn/interpreter/tnn/objseri.h"
#include "onnx2tnn_prefix.h"
#include "onnx_op_converter.h"

#include "onnx.pb.h"
#include "onnx_utility.h"

using namespace std;
using namespace TNN_NS;

const std::string tag = "converter";


struct IndexNode {
    int index;
    onnx::NodeProto* node;
    int mark = 0;
    IndexNode(int index, onnx::NodeProto* node) {
        this->index = index;
        this->node = node;
    }
};

std::vector<int> GetNextIndexNode(std::vector<IndexNode>& index_nodes, int index);
std::vector<int> GetPreviousIndexNode(std::vector<IndexNode>& index_nodes, int index);
int RemoveIndexNode(std::vector<IndexNode> &index_nodes, int index);

class Onnx2TNN {
public:
    Onnx2TNN(std::string onnx_model_path, std::string tnn_proto_path,
             std::string tnn_model_path, InputShapesMap shapes_map = {});
    ~Onnx2TNN();

    int Convert(DataType dataType = DATA_TYPE_FLOAT);

private:
    std::string tnn_proto_path_;
    std::string tnn_model_path_;
    std::string onnx_model_path_;
    onnx::ModelProto* onnx_model_ = nullptr;
    InputShapesMap target_inputs_shape_map_ = {};
    
    int OnnxExtractBlobWeights();
    bool CheckIs3DModel();

    // proto相关
    int TNNWriteProto();

    // model相关
    int TNNWriteModel();

    // onnx node reference
    std::set<std::string> onnx_blob_names_;
    // onnx node reference
    std::map<std::string, int> onnx_node_reference_;

    OnnxNetInfo onnx_net_info_;

protected:
    //clear empty node like const and noop
    int ClearEmptyNode(std::vector<IndexNode>& index_nodes);


    //remove
    int RemoveConsecutiveReshape(onnx::GraphProto* mutable_graph,
                      std::vector<IndexNode>& index_nodes,
                      std::map<std::string, onnx::TensorProto>& weights,
                      std::map<std::string, int>& node_reference,
                      std::set<std::string>& blob_names);
    int RemoveReshape(onnx::GraphProto* mutable_graph,
                      std::vector<IndexNode>& index_nodes,
                      std::map<std::string, onnx::TensorProto>& weights,
                      std::map<std::string, int>& node_reference,
                      std::set<std::string>& blob_names);
    int RemovePool(onnx::GraphProto* mutable_graph,
                   std::vector<IndexNode> & index_nodes,
                   std::map<std::string, onnx::TensorProto>& weights,
                   std::map<std::string, int>& node_reference,
                   std::set<std::string>& blob_names);
    int RemovePad(onnx::GraphProto* mutable_graph,
                  std::vector<IndexNode> & index_nodes,
                  std::map<std::string, onnx::TensorProto>& weights,
                  std::map<std::string, int>& node_reference,
                  std::set<std::string>& blob_names);
    int RemoveUnsqueeze(onnx::GraphProto* mutable_graph,
                  std::vector<IndexNode> & index_nodes,
                  std::map<std::string, onnx::TensorProto>& weights,
                  std::map<std::string, int>& node_reference,
                  std::set<std::string>& blob_names);
    int RemoveExpand(onnx::GraphProto* mutable_graph,
                  std::vector<IndexNode> & index_nodes,
                  std::map<std::string, onnx::TensorProto>& weights,
                  std::map<std::string, int>& node_reference,
                  std::set<std::string>& blob_names);
    int RemoveConcat(onnx::GraphProto* mutable_graph,
                     std::vector<IndexNode> & index_nodes,
                     std::map<std::string, onnx::TensorProto>& weights,
                     std::map<std::string, int>& node_reference,
                     std::set<std::string>& blob_names);
    int RemoveSplitUnsqueezeConcat(onnx::GraphProto* mutable_graph,
                                   std::vector<IndexNode> & index_nodes,
                                   std::map<std::string, onnx::TensorProto>& weights,
                                   std::map<std::string, int>& node_reference,
                                   std::set<std::string>& blob_names);

    int RemoveTranspose(onnx::GraphProto* mutable_graph,
                                  std::vector<IndexNode> & index_nodes,
                                  std::map<std::string, onnx::TensorProto>& weights,
                                  std::map<std::string, int>& node_reference,
                                  std::set<std::string>& blob_names);

    int RemoveImageScaler(onnx::GraphProto* mutable_graph, std::vector<IndexNode>& index_nodes,
                          std::map<std::string, onnx::TensorProto>& weights, std::map<std::string, int>& node_reference,
                          std::set<std::string>& blob_names);

    int RemoveSqueeze(onnx::GraphProto* mutable_graph, std::vector<IndexNode>& index_nodes,
                      std::map<std::string, onnx::TensorProto>& weights, std::map<std::string, int>& node_reference,
                      std::set<std::string>& blob_names);

    int RemoveDropout(onnx::GraphProto* mutable_graph, std::vector<IndexNode>& index_nodes,
                      std::map<std::string, onnx::TensorProto>& weights, std::map<std::string, int>& node_reference,
                      std::set<std::string>& blob_names);
    
    int RemoveReshapeWhere(onnx::GraphProto* mutable_graph, std::vector<IndexNode>& index_nodes,
                      std::map<std::string, onnx::TensorProto>& weights, std::map<std::string, int>& node_reference,
                      std::set<std::string>& blob_names);
    int RemoveIdentity(onnx::GraphProto* mutable_graph,
                       std::vector<IndexNode> & index_nodes,
                       std::map<std::string, onnx::TensorProto>& weights,
                       std::map<std::string, int>& node_reference,
                       std::set<std::string>& blob_names);
protected:
    //fuse
    int FuseLogSigmoid(onnx::GraphProto* mutable_graph,
                       std::vector<IndexNode> & index_nodes,
                       std::map<std::string, onnx::TensorProto>& weights,
                       std::map<std::string, int>& node_reference,
                       std::set<std::string>& blob_names);
    int FuseSoftmax(onnx::GraphProto* mutable_graph,
                    std::vector<IndexNode> & index_nodes,
                    std::map<std::string, onnx::TensorProto>& weights,
                    std::map<std::string, int>& node_reference,
                    std::set<std::string>& blob_names);
    int FuseHardSigmoid(onnx::GraphProto* mutable_graph,
                        std::vector<IndexNode> & index_nodes,
                        std::map<std::string, onnx::TensorProto>& weights,
                        std::map<std::string, int>& node_reference,
                        std::set<std::string>& blob_names);
    //call HardSigmoid before FuseHardSwish
    int FuseHardSwish(onnx::GraphProto* mutable_graph,
                      std::vector<IndexNode> & index_nodes,
                      std::map<std::string, onnx::TensorProto>& weights,
                      std::map<std::string, int>& node_reference,
                      std::set<std::string>& blob_names);
    int FuseGELU(onnx::GraphProto* mutable_graph,
                      std::vector<IndexNode> & index_nodes,
                      std::map<std::string, onnx::TensorProto>& weights,
                      std::map<std::string, int>& node_reference,
                      std::set<std::string>& blob_names);
    int FuseTranspose(onnx::GraphProto* mutable_graph,
                      std::vector<IndexNode> & index_nodes,
                      std::map<std::string, onnx::TensorProto>& weights,
                      std::map<std::string, int>& node_reference,
                      std::set<std::string>& blob_names);
    int FuseShuffleChannel(onnx::GraphProto* mutable_graph,
                           std::vector<IndexNode> & index_nodes,
                           std::map<std::string, onnx::TensorProto>& weights,
                           std::map<std::string, int>& node_reference,
                           std::set<std::string>& blob_names);
    int FuseMatMul(onnx::GraphProto* mutable_graph,
                   std::vector<IndexNode> & index_nodes,
                   std::map<std::string, onnx::TensorProto>& weights,
                   std::map<std::string, int>& node_reference,
                   std::set<std::string>& blob_names);
    int FuseNormalize(onnx::GraphProto* mutable_graph,
                      std::vector<IndexNode> & index_nodes,
                      std::map<std::string, onnx::TensorProto>& weights,
                      std::map<std::string, int>& node_reference,
                      std::set<std::string>& blob_names);
    int FuseFlatten(onnx::GraphProto* mutable_graph,
                    std::vector<IndexNode> & index_nodes,
                    std::map<std::string, onnx::TensorProto>& weights,
                    std::map<std::string, int>& node_reference,
                    std::set<std::string>& blob_names);
    int FusePRelu(onnx::GraphProto* mutable_graph,
                  std::vector<IndexNode> & index_nodes,
                  std::map<std::string, onnx::TensorProto>& weights,
                  std::map<std::string, int>& node_reference,
                  std::set<std::string>& blob_names);
    int FuseLSTM(onnx::GraphProto* mutable_graph,
                  std::vector<IndexNode> & index_nodes,
                  std::map<std::string, onnx::TensorProto>& weights,
                  std::map<std::string, int>& node_reference,
                  std::set<std::string>& blob_names);
    int FuseArgMaxOrMin(onnx::GraphProto* mutable_graph,
                  std::vector<IndexNode> & index_nodes,
                  std::map<std::string, onnx::TensorProto>& weights,
                  std::map<std::string, int>& node_reference,
                  std::set<std::string>& blob_names);
    int FuseBatchNorm(onnx::GraphProto* mutable_graph,
                      std::vector<IndexNode> & index_nodes,
                      std::map<std::string, onnx::TensorProto>& weights,
                      std::map<std::string, int>& node_reference,
                      std::set<std::string>& blob_names);
    int FuseGEMM(onnx::GraphProto* mutable_graph,
                         std::vector<IndexNode> & index_nodes,
                         std::map<std::string, onnx::TensorProto>& weights,
                         std::map<std::string, int>& node_reference,
                         std::set<std::string>& blob_names);
    int FuseSignedMul(onnx::GraphProto* mutable_graph,
                         std::vector<IndexNode> & index_nodes,
                         std::map<std::string, onnx::TensorProto>& weights,
                         std::map<std::string, int>& node_reference,
                         std::set<std::string>& blob_names);

    int FuseDeconv(onnx::GraphProto* mutable_graph,
                   std::vector<IndexNode> & index_nodes,
                   std::map<std::string, onnx::TensorProto>& weights,
                   std::map<std::string, int>& node_reference,
                   std::set<std::string>& blob_names);
    int FuseConv(onnx::GraphProto* mutable_graph,
                   std::vector<IndexNode> & index_nodes,
                   std::map<std::string, onnx::TensorProto>& weights,
                   std::map<std::string, int>& node_reference,
                   std::set<std::string>& blob_names);
    int FuseHDRGuide(onnx::GraphProto* mutable_graph,
                     std::vector<IndexNode> & index_nodes,
                     std::map<std::string, onnx::TensorProto>& weights,
                     std::map<std::string, int>& node_reference,
                     std::set<std::string>& blob_names);
    int FuseDepthToSpace(onnx::GraphProto* mutable_graph,
                         std::vector<IndexNode> & index_nodes,
                         std::map<std::string, onnx::TensorProto>& weights,
                         std::map<std::string, int>& node_reference,
                         std::set<std::string>& blob_names);

    int FuseGlobalAveragePool(onnx::GraphProto* mutable_graph,
                         std::vector<IndexNode> & index_nodes,
                         std::map<std::string, onnx::TensorProto>& weights,
                         std::map<std::string, int>& node_reference,
                         std::set<std::string>& blob_names);

    int FuseLayerNormalization(onnx::GraphProto* mutable_graph,
                                  std::vector<IndexNode> & index_nodes,
                                  std::map<std::string, onnx::TensorProto>& weights,
                                  std::map<std::string, int>& node_reference,
                                  std::set<std::string>& blob_names);
    int FuseInstanceNormalization(onnx::GraphProto* mutable_graph,
                                  std::vector<IndexNode> & index_nodes,
                                  std::map<std::string, onnx::TensorProto>& weights,
                                  std::map<std::string, int>& node_reference,
                                  std::set<std::string>& blob_names);
    int FuseGroupNormalization(onnx::GraphProto* mutable_graph,
                                  std::vector<IndexNode> & index_nodes,
                                  std::map<std::string, onnx::TensorProto>& weights,
                                  std::map<std::string, int>& node_reference,
                                  std::set<std::string>& blob_names);
    
    int FusePooling(onnx::GraphProto* mutable_graph, std::vector<IndexNode>& index_nodes,
                    std::map<std::string, onnx::TensorProto>& weights, std::map<std::string, int>& node_reference,
                    std::set<std::string>& blob_names);
    int FuseRelu6(onnx::GraphProto* mutable_graph, std::vector<IndexNode>& index_nodes,
                    std::map<std::string, onnx::TensorProto>& weights, std::map<std::string, int>& node_reference,
                    std::set<std::string>& blob_names);
    int FuseSpaceToDepth(onnx::GraphProto* mutable_graph, std::vector<IndexNode>& index_nodes,
                         std::map<std::string, onnx::TensorProto>& weights,
                         std::map<std::string, int>& node_reference, std::set<std::string>& blob_names);
    int FuseHistogram(onnx::GraphProto* mutable_graph, std::vector<IndexNode>& index_nodes,
                    std::map<std::string, onnx::TensorProto>& weights, std::map<std::string, int>& node_reference,
                    std::set<std::string>& blob_names);

    int FuseClip(onnx::GraphProto* mutable_graph, std::vector<IndexNode>& index_nodes,
                 std::map<std::string, onnx::TensorProto>& weights, std::map<std::string, int>& node_reference,
                 std::set<std::string>& blob_names);

    int FuseSwish(onnx::GraphProto* mutable_graph, std::vector<IndexNode>& index_nodes,
                 std::map<std::string, onnx::TensorProto>& weights, std::map<std::string, int>& node_reference,
                 std::set<std::string>& blob_names);

    int FuseScatterElements(onnx::GraphProto* mutable_graph, std::vector<IndexNode>& index_nodes,
                 std::map<std::string, onnx::TensorProto>& weights, std::map<std::string, int>& node_reference,
                 std::set<std::string>& blob_names);

protected:
    //transfer
    int TransferReduceMax(onnx::GraphProto* mutable_graph,
                           std::vector<IndexNode> & index_nodes,
                           std::map<std::string, onnx::TensorProto>& weights,
                           std::map<std::string, int>& node_reference,
                           std::set<std::string>& blob_names);
    
    int TransferGlobalMaxPool(onnx::GraphProto* mutable_graph, 
                              std::vector<IndexNode>& index_nodes,
                              std::map<std::string, onnx::TensorProto>& weights,
                              std::map<std::string, int>& node_reference,
                              std::set<std::string>& blob_names);
    int TransferGroupNormalization(onnx::GraphProto* mutable_graph,
                              std::vector<IndexNode>& index_nodes,
                              std::map<std::string, onnx::TensorProto>& weights,
                              std::map<std::string, int>& node_reference,
                              std::set<std::string>& blob_names);
    int TransferInverse(onnx::GraphProto* mutable_graph,
                              std::vector<IndexNode>& index_nodes,
                              std::map<std::string, onnx::TensorProto>& weights,
                              std::map<std::string, int>& node_reference,
                              std::set<std::string>& blob_names);
    int TransferGridSample(onnx::GraphProto* mutable_graph,
                              std::vector<IndexNode>& index_nodes,
                              std::map<std::string, onnx::TensorProto>& weights,
                              std::map<std::string, int>& node_reference,
                              std::set<std::string>& blob_names);
    
    int TransferInputName(onnx::GraphProto* mutable_graph);

    int TransferSplit(onnx::GraphProto* mutable_graph,
                      std::vector<IndexNode> & index_nodes,
                      std::map<std::string, onnx::TensorProto>& weights,
                      std::map<std::string, int>& node_reference,
                      std::set<std::string>& blob_names);

    int TransferConcat(onnx::GraphProto* mutable_graph,
                      std::vector<IndexNode> & index_nodes,
                      std::map<std::string, onnx::TensorProto>& weights,
                      std::map<std::string, int>& node_reference,
                      std::set<std::string>& blob_names);


};

std::string get_backtrack();

#endif /* onnx2tnn_hpp */
