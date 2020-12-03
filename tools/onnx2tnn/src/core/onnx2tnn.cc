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

#include "onnx2tnn.h"

#include <math.h>
#include <stdio.h>
#include <execinfo.h>

#include <set>
#include <exception>

#include "onnx2tnn_prefix.h"

using namespace std;

std::vector<int> GetNextIndexNode(std::vector<IndexNode>& index_nodes, int index) {
    auto node = index_nodes[index].node;
    std::vector<int> next_indexes;

    for (int ii = index + 1; ii < index_nodes.size(); ii++) {
        auto item    = index_nodes[ii].node;
        bool is_next = false;
        for (auto output : node->output()) {
            for (auto input : item->input()) {
                if (output == input) {
                    next_indexes.push_back(ii);
                    is_next = true;
                    break;
                }
            }
            if (is_next) {
                break;
            }
        }
    }
    return next_indexes;
}

int RemoveIndexNode(std::vector<IndexNode>& index_nodes, int index) {
    auto node                     = index_nodes[index].node;
    auto node_input               = node->input(0);
    std::vector<int> next_indexes = GetNextIndexNode(index_nodes, index);
    for (auto index : next_indexes) {
        auto next_node = index_nodes[index].node;
        for (int output_index = 0; output_index < node->output_size(); output_index++) {
            for (int ii = 0; ii < next_node->input_size(); ii++) {
                if (node->output(output_index) == next_node->input(ii)) {
                    next_node->set_input(ii, node_input);
                }
            }
        }
    }
    return 0;
}

Onnx2TNN::Onnx2TNN(std::string onnx_model_path, std::string tnn_proto_path, std::string tnn_model_path) {
    tnn_proto_path_ = tnn_proto_path;
    tnn_model_path_ = tnn_model_path;

    onnx_model_path_ = onnx_model_path;
}

Onnx2TNN::~Onnx2TNN() {
    if (!onnx_model_) {
        delete onnx_model_;
    }
}

bool Onnx2TNN::CheckIs3DModel() {
    const onnx::GraphProto& graph = onnx_model_->graph();
    //写入每层的权值
    for (int i = 0; i < graph.node_size(); i++) {
        const onnx::NodeProto& node = graph.node(i);
        const std::string& onnx_op  = node.op_type();
        if (onnx_op == "Conv") {
            std::vector<int64_t> kernel_shape = get_node_attr_ai(node, "kernel_shape");
            return kernel_shape.size() == 3;
        }
    }
    return false;
}

int Onnx2TNN::Convert(DataType dataType) {
    int ret = 0;

    //加载onnx模型
    if (!onnx_model_) {
        onnx::ModelProto* onnx_model = new onnx::ModelProto();
        ret                          = read_proto_from_binary(onnx_model_path_.c_str(), onnx_model);

        if (ret != 0) {
            delete onnx_model;

            LOGE("read_proto_from_binary failed, path:%s\n", onnx_model_path_.c_str());
            return ret;
        }

        onnx_model_ = onnx_model;
    }

    //提取onnx的blob和weights信息
    ret = OnnxExtractBlobWeights();
    if (ret != 0) {
        LOGE("OnnxExtractBlobWeights failed");
        return ret;
    }

    onnx_net_info_.data_type   = dataType;
    onnx_net_info_.is_3D_model = CheckIs3DModel();
    onnx_net_info_.opset       = onnx_model_->opset_import(0).version();

    ret = TNNWriteProto();
    if (ret != 0) {
        LOGE("TNNWriteProto failed");
        return ret;
    }

    ret = TNNWriteModel();
    if (ret != 0) {
        LOGE("TNNWriteModel failed");
        return ret;
    }

    return ret;
}

int Onnx2TNN::TNNWriteProto() {
    int ret          = 0;
    FILE* file_proto = nullptr;

    do {
        file_proto = fopen(tnn_proto_path_.c_str(), "w");
        if (!file_proto) {
            LOGE("fopen proto file failed, path:%s\n", tnn_proto_path_.c_str());
            break;
        }

        const onnx::GraphProto& graph = onnx_model_->graph();

        ostringstream proto_net_info;
        {
            // line 1
            proto_net_info << "\"1 " << (int)onnx_blob_names_.size() << " 1 " << g_version_magic_number_tnn << " ,\""
                           << endl;

            // line 2, 输入blob
            {
                int intput_blob_count                          = 0;
                std::vector<onnx::ValueInfoProto*> input_blobs = std::vector<onnx::ValueInfoProto*>();
                for (int j = 0; j < graph.input_size(); j++) {
                    const std::string& input_name = graph.input(j).name();

                    // check weight
                    if (onnx_net_info_.weights_map.find(input_name) != onnx_net_info_.weights_map.end())
                        continue;

                    // split the input
                    if (onnx_node_reference_.size() > 0) {
                        if (onnx_node_reference_.find(input_name) != onnx_node_reference_.end()) {
                            int refcount = onnx_node_reference_[input_name];
                            if (refcount < 1) {
                                continue;
                            }
                        }
                    }

                    // check is unused input
                    bool is_used_input = false;
                    for (int iz = 0; iz < graph.node_size() && !is_used_input; iz++) {
                        const onnx::NodeProto& node = graph.node(iz);
                        for (int nz = 0; nz < node.input_size(); nz++) {
                            if (input_name == node.input(nz)) {
                                is_used_input = true;
                                break;
                            }
                        }
                    }

                    if (!is_used_input) {
                        continue;
                    }

                    intput_blob_count++;
                    input_blobs.push_back((onnx::ValueInfoProto*)(&(graph.input(j))));
                }

                if (intput_blob_count == 0) {
                    LOGE("invalid input blob count(must >= 1): %d\n", intput_blob_count);
                    assert(0);
                    break;
                }

                proto_net_info << "\"";
                for (int ii = 0; ii < intput_blob_count; ii++) {
                    onnx::ValueInfoProto* input_blob = input_blobs[ii];
                    int shape_n = 0, shape_c = 0, shape_d = 0, shape_h = 0, shape_w = 0;
                    const ::onnx::TensorShapeProto& input_blob_shape = input_blob->type().tensor_type().shape();
                    if (input_blob_shape.dim_size() == 4) {
                        shape_n = (int)input_blob_shape.dim(0).dim_value();
                        shape_c = (int)input_blob_shape.dim(1).dim_value();
                        shape_h = (int)input_blob_shape.dim(2).dim_value();
                        shape_w = (int)input_blob_shape.dim(3).dim_value();
                        shape_n = std::max(shape_n, 1);

                        LOGD(
                            "input_blob %d(%s) channel order:%d(%s) %d(%s) "
                            "%d(%s) %d(%s)\n",
                            ii, input_blob->name().c_str(), shape_n, input_blob_shape.dim(0).denotation().c_str(),
                            shape_c, input_blob_shape.dim(1).denotation().c_str(), shape_h,
                            input_blob_shape.dim(2).denotation().c_str(), shape_w,
                            input_blob_shape.dim(3).denotation().c_str());

                        if (shape_c <= 0 || shape_h <= 0 || shape_w <= 0) {
                            LOGE("intput shape is invalid\n");
                            assert(0);
                        }

                        proto_net_info << input_blob->name() << " " << shape_n << " " << shape_c << " " << shape_h
                                       << " " << shape_w << " ";
                    } else if (input_blob_shape.dim_size() == 5) {
                        shape_n = (int)input_blob_shape.dim(0).dim_value();
                        shape_c = (int)input_blob_shape.dim(1).dim_value();
                        shape_d = (int)input_blob_shape.dim(2).dim_value();
                        shape_h = (int)input_blob_shape.dim(3).dim_value();
                        shape_w = (int)input_blob_shape.dim(4).dim_value();
                        LOGD(
                            "input_blob %d(%s) channel order:%d(%s) %d(%s) "
                            "%d(%s) %d(%s) %d(%s)\n",
                            ii, input_blob->name().c_str(), shape_n, input_blob_shape.dim(0).denotation().c_str(),
                            shape_c, input_blob_shape.dim(1).denotation().c_str(), shape_d,
                            input_blob_shape.dim(2).denotation().c_str(), shape_h,
                            input_blob_shape.dim(3).denotation().c_str(), shape_w,
                            input_blob_shape.dim(4).denotation().c_str());

                        proto_net_info << input_blob->name() << " " << shape_n << " " << shape_c << " " << shape_d
                                       << " " << shape_h << " " << shape_w << " ";
                    } else {
                        LOGE("input_blob_shape invalid\n");
                        assert(0);
                        break;
                    }

                    if (intput_blob_count > 1 && ii != intput_blob_count - 1) {
                        proto_net_info << ": ";
                    }
                }
                proto_net_info << ",\"" << endl;
            }

            // line 3, 所有blob
            {
                proto_net_info << "\" ";
                for (auto item = onnx_blob_names_.begin(); item != onnx_blob_names_.end(); item++) {
                    proto_net_info << *item << " ";
                }
                proto_net_info << ",\"" << endl;
            }

            // line 4, 输出blob
            {
                int output_blob_count             = 0;
                onnx::ValueInfoProto* output_blob = nullptr;

                proto_net_info << "\"";
                for (int j = 0; j < graph.output_size(); j++) {
                    const std::string& output_name = graph.output(j).name();

                    // check weight
                    if (onnx_net_info_.weights_map.find(output_name) != onnx_net_info_.weights_map.end())
                        continue;

                    output_blob_count++;
                    output_blob = (onnx::ValueInfoProto*)(&(graph.output(j)));
                    proto_net_info << output_blob->name() << " ";
                }
                proto_net_info << ",\"" << endl;

                if (output_blob_count <= 0) {
                    LOGE("invalid output blob count(must = 1): %d\n", output_blob_count);
                    assert(0);
                    break;
                }
            }
        }

        int layer_count = 0;
        ostringstream proto_layers;
        {
            for (int i = 0; i < graph.node_size(); i++) {
                onnx::NodeProto& node      = (onnx::NodeProto&)graph.node(i);
                const std::string& onnx_op = node.op_type();

                if (onnx_op == k_tnn_noop_type || onnx_op == "Constant") {
                    continue;
                }

                auto op_converter = OnnxOpConverterManager::Shared()->GetOnnxOpConverter(onnx_op);
                if (!op_converter) {
                    LOGE("error::op convert failed onnx:%s\n", onnx_op.c_str());
                    assert(0);
                } else {
                    LOGD("node:%s onnx:%s -> tnn:%s\n", node.output(0).c_str(), onnx_op.c_str(),
                         op_converter->TNNOpType(node, onnx_net_info_).c_str());
                }

                auto op_proto = op_converter->TNNLayerProto(node, onnx_net_info_);
                proto_layers << op_proto << endl;
                layer_count++;
            }
        }

        fprintf(file_proto, "%s", proto_net_info.str().c_str());
        // line 5, 层数 TODO
        fprintf(file_proto, "\" %d ,\"\n", layer_count);
        fprintf(file_proto, "%s", proto_layers.str().c_str());

        // LOGE("%s", proto_net_info.str().c_str());
        // // line 5, 层数 TODO
        // LOGE("\" %d ,\"\n", layer_count);
        // LOGE("%s", proto_layers.str().c_str());

    } while (0);

    if (file_proto) {
        fclose(file_proto);
    }
    return ret;
}

int Onnx2TNN::OnnxExtractBlobWeights() {
    if (!onnx_model_) {
        LOGE("onnx_model is nil");
        return -1;
    }

    const onnx::GraphProto& graph   = onnx_model_->graph();
    onnx::GraphProto* mutable_graph = onnx_model_->mutable_graph();
    TransferInputName(mutable_graph);

    int node_count = graph.node_size();

    // node reference
    std::map<std::string, int> node_reference;
    std::map<std::string, std::vector<int>> follow_up_node_ids;
    std::map<std::string, int> node_name_to_node_id;

    //去除常量node，便于fuse判断pattern
    std::vector<IndexNode> index_nodes;
    for (int i = 0; i < node_count; i++) {
        auto node = mutable_graph->mutable_node(i);
        index_nodes.push_back(IndexNode(i, node));
    }
    ClearEmptyNode(index_nodes);

    // weight node and weight reshape node
    TensorProtoMap weights;
    TensorShapeMap weight_shapes;

    for (int j = 0; j < graph.initializer_size(); j++) {
        const onnx::TensorProto& initializer = graph.initializer(j);

        // fprintf(stderr, "weight = %s\n", initializer.name().c_str());
        LOGD("weight = %s\n", initializer.name().c_str());

        weights[initializer.name()] = initializer;
    }

    for (int j = 0; j < graph.value_info_size(); j++) {
        const onnx::TensorShapeProto& shape_info = graph.value_info(j).type().tensor_type().shape();

        // fprintf(stderr, "value_info dim_size = %d\n", shape_info.dim_size());
        LOGD("value_info dim_size = %d\n", shape_info.dim_size());

        weight_shapes[graph.value_info(j).name()] = shape_info;
    }

    // global definition line
    // [layer count] [blob count]
    std::set<std::string> blob_names;
    for (int i = 0; i < node_count; i++) {
        const onnx::NodeProto& node = graph.node(i);

        const std::string& onnx_op = node.op_type();

        std::string name = node.name();
        if (name.empty()) {
            name = node.output(0);
        }

        if (onnx_op == "Constant") {
            onnx::TensorProto tensor = get_node_attr_tensor(node, "value");
            weights[node.output(0)]  = tensor;
            LOGD("const node = %s\n", name.c_str());
            continue;
        } else if (onnx_op == "Cast") {
        } else if (onnx_op == "Reshape") {
            if (node.input_size() == 1) {
                const std::string& input_name = node.input(0);

                // check weight
                if (weights.find(input_name) != weights.end()) {
                    weights[node.output(0)] = weights[input_name];
                    continue;
                }
            } else if (node.input_size() == 2) {
                // opset 5
                const std::string& input_name     = node.input(1);
                const onnx::TensorProto& shape_tp = weights[node.input(1)];
                const int64_t* shape_data         = shape_tp.int64_data().data();

                // check weight
                if (weights.find(input_name) != weights.end()) {
                    /*
                    weights[node.output(0)] = weights[input_name];

                    // set weight shape directly
                    const onnx::TensorProto& shape_tp = weights[node.input(1)];
                    const int64_t* shape_data = shape_tp.int64_data().data();

                    weights[node.output(0)].clear_dims();
                    for (int j = 0; j < shape_tp.int64_data_size(); j++) {
                        weights[node.output(0)].add_dims(shape_data[j]);
                    }
                    */

                } else {
                    assert(0);
                    // DLog("Reshape %s can't find input 0 weights. set shape as const values\n",
                    // node.input(1).c_str()); onnx::TensorProto tensor_shape; tensor_shape.set_data_type(7);
                    // tensor_shape.add_int64_data(0);
                    // tensor_shape.add_int64_data(-1);
                    // tensor_shape.add_int64_data(1);
                    // tensor_shape.add_int64_data(1);
                    // weights[node.input(1)] = tensor_shape;
                    // for(int i=0;i<4;i++) {
                    // weights[node.output(0)].add_dims(4);
                    // }
                }
            }
        }

        for (int j = 0; j < (int)node.input_size(); j++) {
            const std::string& input_name = node.input(j);

            // check weight
            if (weights.find(input_name) != weights.end()) {
                continue;
            }

            blob_names.insert(input_name);

            if (node_reference.find(input_name) == node_reference.end()) {
                node_reference[input_name] = 1;
            } else {
                node_reference[input_name] = node_reference[input_name] + 1;
            }

            // 记录依赖 node name 的节点列表
            {
                std::vector<int> node_list;
                if (follow_up_node_ids.find(input_name) != follow_up_node_ids.end()) {
                    node_list = follow_up_node_ids[input_name];
                }
                node_list.push_back(i);
                follow_up_node_ids[input_name] = node_list;
            }
        }

        for (int j = 0; j < (int)node.output_size(); j++) {
            const std::string& output_name = node.output(j);

            blob_names.insert(output_name);

            // 记录node name 对应的node id
            {
                // 每个node name只应该出现一次
                if (node_name_to_node_id.find(output_name) != node_name_to_node_id.end()) {
                    assert(0);
                }
                node_name_to_node_id[output_name] = i;
            }

            // 简化代码逻辑， Dropout 只取0号output，原因暂不清楚
            if (onnx_op == "Dropout") {
                break;
            }
        }
    }

    // include Input node
    int input_node_count = 0;
    for (int j = 0; j < graph.input_size(); j++) {
        const std::string& input_name = graph.input(j).name();

        // check weight
        if (weights.find(input_name) != weights.end())
            continue;

        blob_names.insert(input_name);

        input_node_count++;
    }
    onnx_net_info_.weights_map       = weights;
    onnx_net_info_.weights_shape_map = weight_shapes;

    // onnx_op remove
    RemovePad(mutable_graph, index_nodes, weights, node_reference, blob_names);
    RemoveExpand(mutable_graph, index_nodes, weights, node_reference, blob_names);
    RemovePool(mutable_graph, index_nodes, weights, node_reference, blob_names);
    RemoveConcat(mutable_graph, index_nodes, weights, node_reference, blob_names);
    RemoveConsecutiveReshape(mutable_graph, index_nodes, weights, node_reference, blob_names);
    // RemoveReshape(mutable_graph, index_nodes, weights, node_reference, blob_names);
    FuseShuffleChannel(mutable_graph, index_nodes, weights, node_reference, blob_names);
    RemoveSplitUnsqueezeConcat(mutable_graph, index_nodes, weights, node_reference, blob_names);
    RemoveSqueeze(mutable_graph, index_nodes, weights, node_reference, blob_names);
    RemoveUnsqueeze(mutable_graph, index_nodes, weights, node_reference, blob_names);
    RemoveDropout(mutable_graph, index_nodes, weights, node_reference, blob_names);
    RemoveImageScaler(mutable_graph, index_nodes, weights, node_reference, blob_names);
    FuseHDRGuide(mutable_graph, index_nodes, weights, node_reference, blob_names);
    // op transfer
    TransferReduceMax(mutable_graph, index_nodes, weights, node_reference, blob_names);
    TransferGlobalMaxPool(mutable_graph, index_nodes, weights, node_reference, blob_names);

    // onnx_op chain fusion
    FuseMatMul(mutable_graph, index_nodes, weights, node_reference, blob_names);
    // FuseShuffleChannel(mutable_graph, index_nodes, weights, node_reference, blob_names);
    FuseLogSigmoid(mutable_graph, index_nodes, weights, node_reference, blob_names);
    FuseSoftmax(mutable_graph, index_nodes, weights, node_reference, blob_names);
    FuseHardSigmoid(mutable_graph, index_nodes, weights, node_reference, blob_names);
    FuseHardSwish(mutable_graph, index_nodes, weights, node_reference, blob_names);
    FuseBatchNorm(mutable_graph, index_nodes, weights, node_reference, blob_names);
    FusePRelu(mutable_graph, index_nodes, weights, node_reference, blob_names);
    FuseNormalize(mutable_graph, index_nodes, weights, node_reference, blob_names);
    FuseFlatten(mutable_graph, index_nodes, weights, node_reference, blob_names);

    FuseSignedMul(mutable_graph, index_nodes, weights, node_reference, blob_names);

    FuseGEMM(mutable_graph, index_nodes, weights, node_reference, blob_names);
    FuseDeconv(mutable_graph, index_nodes, weights, node_reference, blob_names);
    FuseConv(mutable_graph, index_nodes, weights, node_reference, blob_names);

    FuseDepthToSpace(mutable_graph, index_nodes, weights, node_reference, blob_names);
    FuseGlobalAveragePool(mutable_graph, index_nodes, weights, node_reference, blob_names);
    FuseInstanceNormalization(mutable_graph, index_nodes, weights, node_reference, blob_names);
    FusePooling(mutable_graph, index_nodes, weights, node_reference, blob_names);
    FuseRelu6(mutable_graph, index_nodes, weights, node_reference, blob_names);
    FuseSpaceToDepth(mutable_graph, index_nodes, weights, node_reference, blob_names);

#ifdef PROCESS_TF
    TransferSplit(mutable_graph, index_nodes, weights, node_reference, blob_names);
    TransferConcat(mutable_graph, index_nodes, weights, node_reference, blob_names);
    RemoveTranspose(mutable_graph, index_nodes, weights, node_reference, blob_names);
#endif
    //    //onnx_op split
    //    for (int i = 0; i < node_count; i++) {
    //        onnx::NodeProto* node = mutable_graph->mutable_node(i);
    //
    //        // ConvTranspose => ConvTranspose - Pad
    //        do {
    //            if (node->op_type() == "ConvTranspose") {
    //                onnx::NodeProto* node_deconv = node;
    //                std::vector<int64_t> output_pads = get_node_attr_ai(*node_deconv, "output_padding");
    //
    //                bool all_zero = true;
    //                for (auto iter : output_pads) {
    //                    if (iter != 0) {
    //                        all_zero = false;
    //                        break;
    //                    }
    //                }
    //                BREAK_IF(all_zero);
    //
    //                reduced_node_count -= 1;
    //                i += 1;
    //            }
    //        } while (0);
    //    }

    // remove node_reference entry with reference equals to one
    int splitncnn_blob_count                = 0;
    std::map<std::string, int>::iterator it = node_reference.begin();
    while (it != node_reference.end()) {
        if (it->second == 1) {
            node_reference.erase(it++);
        } else {
            splitncnn_blob_count += it->second;
            //             fprintf(stderr, "%s %d\n", it->first.c_str(),
            //             it->second);
            ++it;
        }
    }

    onnx_blob_names_                 = blob_names;
    onnx_node_reference_             = node_reference;
    onnx_net_info_.weights_map       = weights;
    onnx_net_info_.weights_shape_map = weight_shapes;
    return 0;
}

int Onnx2TNN::ClearEmptyNode(std::vector<IndexNode>& index_nodes) {
    std::vector<IndexNode> nodes;
    for (auto item : index_nodes) {
        if (item.node->op_type() == "Constant") {
            continue;
        } else if (item.node->op_type() == k_tnn_noop_type) {
            continue;
        }
        nodes.push_back(item);
    }
    index_nodes = nodes;
    return 0;
}

std::string get_backtrack() {
    const int MAX_SIZE = 10;
    std::string backtrace_str;
    char **strings = nullptr;
    void *array[MAX_SIZE] = {0};
    size_t size = backtrace(array, MAX_SIZE);
    strings = backtrace_symbols(array, size);
    for (size_t i = 0; i < size; i++)
        backtrace_str += std::string(strings[i]) + std::string("\n");
    free(strings);
    return backtrace_str;
}